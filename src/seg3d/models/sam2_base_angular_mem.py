import torch
from torch.utils.checkpoint import checkpoint

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import get_1d_sine_pe

from seg3d.utils.view_sampling import compute_angular_distance

class MultiViewSAM2Base(SAM2Base):
    """
    Extension of SAM2Base that supports multi-view / multi-camera memory
    using angular view proximity between frames.
    """
  
    def forward_image(self, inputs: dict):
        """Get the image features on the input batch (multi-modal: normal + point)."""
        # NOTE: image_encoder should be GeoSAM2MultimodalEncoder that accepts (normal, point)
        backbone_out = self.image_encoder(inputs["normal"], inputs["point"])
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out
    
    # ---- 1. Overlap rank map -------------------------------------------------
    def _create_view_proximity_rank_map(
        self,
        frame_idx,
        output_dict,
        view_dirs,
    ):
        """
        Calculates spatial proximity between views (using angular distance)
        and assigns a synthetic rank index (t_proximity) where the index distance
        from frame_idx reflects spatial closeness.

        Returns: {mem_type: {t_proximity_idx: mem_idx}}
        """
        # Convert view_dirs to numpy array if it's a torch tensor
        if isinstance(view_dirs, torch.Tensor):
            view_dirs = view_dirs.cpu().numpy()
        
        curr_view_dir = view_dirs[frame_idx]
        past_candidates = []    # mem_idx < frame_idx
        future_candidates = []  # mem_idx > frame_idx 
        
        # Combine Cond and Non-Cond outputs for initial scoring
        all_outputs = {}
        all_outputs.update(output_dict.get("cond_frame_outputs", {}))
        all_outputs.update(output_dict.get("non_cond_frame_outputs", {})) 
          
        for mem_idx, mem_out in all_outputs.items():
            if mem_idx == frame_idx:
                continue
            if mem_out.get("maskmem_features", None) is None:
                continue
            
            # Compute Angular Distance (Centralized calculation)
            angle_dist = compute_angular_distance(
                view_dirs[mem_idx], curr_view_dir
            )
                
            mem_type = "cond_frame_outputs" if mem_idx in output_dict["cond_frame_outputs"] else "non_cond_frame_outputs"
            
            # Store: (angular_distance, mem_idx, mem_type)
            candidate = (angle_dist, mem_idx, mem_type)
            
            # Separate based on consistency (before/after target frame)
            if mem_idx < frame_idx:
                past_candidates.append(candidate)
            else:
                future_candidates.append(candidate)                                 
        
        # Sort by Angular Distance (Lowest First)
        past_candidates.sort(key=lambda x: x[0], reverse=False)
        future_candidates.sort(key=lambda x: x[0], reverse=False)
        
        # Assign View Proximity Rank Index (t_proximity) and Create Map
        view_proximity_rank_map = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        
        # Past Frames: Assign index FRAME_IDX - RANK (Lowest angular distance gets rank 1)
        for rank, (angle_dist, mem_idx, mem_type) in enumerate(past_candidates, start=1):
            t_proximity_idx = frame_idx - rank
            view_proximity_rank_map[mem_type][t_proximity_idx] = mem_idx
            
        # Future Frames: Assign index FRAME_IDX + RANK
        for rank, (angle_dist, mem_idx, mem_type) in enumerate(future_candidates, start=1):
            t_proximity_idx = frame_idx + rank
            view_proximity_rank_map[mem_type][t_proximity_idx] = mem_idx
              
        return view_proximity_rank_map
    
    # ---- 2. Select cond frames based on overlap rank -------------------------
    def select_cond_frames_by_view_proximity_rank(
        self,
        frame_idx: int,
        cond_frame_outputs: dict,
        cond_map: dict,
    ): 
        """
        Selects conditioning frames using a hybrid approach:
        (a, b) Temporal criteria (using true mem_idx)
        (c) Spatial criteria (using t_proximity rank)
        
        This function avoids redundant view proximity calculations.
        
        Returns: (selected_outputs, unselected_outputs, selected_mapping, unselected_mapping)
        """  
        max_cond_frame_num = self.max_cond_frames_in_attn
        # Use the Cond part of the view_proximity_rank_map for efficient ranking
        
        # Create the reverse map: mem_idx -> t_proximity_idx, for O(1) rank lookups
        #reverse_cond_map = {v: k for k, v in cond_map.items()}
        
        if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
            selected_outputs = cond_frame_outputs
            unselected_outputs = {}
        else:
            assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
            selected_outputs = {}

            # the closest conditioning frame before `frame_idx` (if any)
            idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
            if idx_before is not None:
                selected_outputs[idx_before] = cond_frame_outputs[idx_before]

            # the closest conditioning frame after `frame_idx` (if any)
            idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
            if idx_after is not None:
                selected_outputs[idx_after] = cond_frame_outputs[idx_after]
            
            # (c) Spatial Selection (Uses View Proximity Rank Index) ---
            num_remain = max_cond_frame_num - len(selected_outputs)
        
            if num_remain > 0:
                # 1. Identify all t_proximity_idx keys that correspond to UNSELECTED mem_idx
                unselected_t_proximity_indices = (
                    t_proximity_idx 
                    for t_proximity_idx, mem_idx in cond_map.items() 
                    if mem_idx not in selected_outputs
                )

                # 2. Sort the t_proximity_idx keys by their distance from frame_idx (Lowest distance = Highest proximity rank)
                inds_remain_t_proximity = sorted(
                    unselected_t_proximity_indices,
                    key=lambda t: abs(t - frame_idx),
                )[:num_remain]

                # 3. Update selected_outputs using the t_proximity_idx -> mem_idx map
                newly_selected = [cond_map[t] for t in inds_remain_t_proximity]
                
                selected_outputs.update(
                    (mem_idx, cond_frame_outputs[mem_idx]) 
                    for mem_idx in newly_selected
                )
                
            # 4. Finalize unselected
            unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}
        
        selected_cond_mapping = {}
        unselected_cond_mapping = {}

        for t_proximity_idx, mem_idx in cond_map.items():
            if mem_idx in selected_outputs:
                selected_cond_mapping[t_proximity_idx] = mem_idx
            else:
                unselected_cond_mapping[t_proximity_idx] = mem_idx

        return (
            selected_outputs,
            unselected_outputs,
            selected_cond_mapping,
            unselected_cond_mapping,
        )    
        
   # ---- 3. Multi-view memory fusion ----------------------------------------
    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        view_dirs,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        
    ):  
        
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            view_proximity_rank_map = self._create_view_proximity_rank_map(
                frame_idx, output_dict, view_dirs
            )
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            
            (
            selected_cond_outputs,
            unselected_cond_outputs,
            selected_cond_mapping, # t_proximity -> mem_idx for selected
            unselected_cond_mapping, # t_proximity -> mem_idx for unselected
            ) = self.select_cond_frames_by_view_proximity_rank(
                frame_idx, 
                output_dict["cond_frame_outputs"],
                view_proximity_rank_map["cond_frame_outputs"]
            )
            
            # A. Add selected Conditioning Frames (t_pos=0)
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            unselected_and_noncond_map = {}
            unselected_and_noncond_map.update(unselected_cond_mapping)
            unselected_and_noncond_map.update(view_proximity_rank_map["non_cond_frame_outputs"])

            for t_pos in range(1, self.num_maskmem):
                t_dist = self.num_maskmem - t_pos  # how many frames before current frame
               
                if not track_in_reverse:
                    prev_proximity_idx = frame_idx - t_dist
                else:
                    prev_proximity_idx = frame_idx + t_dist
                
                prev_mem_idx = unselected_and_noncond_map.get(prev_proximity_idx, None)
                out = output_dict["non_cond_frame_outputs"].get(prev_mem_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_mem_idx, None)
                
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)
            
            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                
                reverse_selected_cond_mapping = {v: k for k, v in selected_cond_mapping.items()}
                
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        reverse_selected_cond_mapping[t]: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = {
                        reverse_selected_cond_mapping[t]: out
                        for t, out in selected_cond_outputs.items()
                    }
                    
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - proximity_t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - proximity_t)
                        ),
                        out["obj_ptr"],
                    )
                    for proximity_t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    proximity_idx = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if proximity_idx < 0 or (num_frames is not None and proximity_idx >= num_frames):
                        break
                    mem_idx = unselected_and_noncond_map.get(proximity_idx, None)
                    out = output_dict["non_cond_frame_outputs"].get(mem_idx, None)
                    if out is None:
                        out = unselected_cond_outputs.get(mem_idx, None)    
                    
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        # Forward through memory attention with gradient checkpointing to save memory
        # Note: checkpoint doesn't handle list inputs well, so we extract the tensor
        # and re-wrap it for memory_attention
        if self.training:
            # Capture the list structure for the closure
            is_list = isinstance(current_vision_feats, list)
            if is_list:
                # Extract the last tensor (current frame) for checkpointing
                curr_tensor = current_vision_feats[-1]
                # Store the full list structure
                curr_list_structure = current_vision_feats
            else:
                curr_tensor = current_vision_feats
                curr_list_structure = None
            
            def _memory_attention_forward(curr_tensor, curr_pos, memory, memory_pos, num_obj_ptr_tokens):
                # Re-wrap as list for memory_attention
                if is_list:
                    # Replace the last element with the checkpointed tensor
                    curr_list = curr_list_structure[:-1] + [curr_tensor]
                else:
                    curr_list = [curr_tensor]
                return self.memory_attention(
                    curr=curr_list,
                    curr_pos=curr_pos,
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=num_obj_ptr_tokens,
                )
            
            pix_feat_with_mem = checkpoint(
                _memory_attention_forward,
                curr_tensor,
                current_vision_pos_embeds,
                memory,
                memory_pos_embed,
                num_obj_ptr_tokens,
                use_reentrant=False,
            )
        else:
            pix_feat_with_mem = self.memory_attention(
                curr=current_vision_feats,
                curr_pos=current_vision_pos_embeds,
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
        # reshape the output (HW)BC => BCHW
        # memory_attention returns [seq_len, batch, dim] format
        # Expected: seq_len = H*W, batch = B, dim = C
        expected_seq_len = H * W
        actual_shape = pix_feat_with_mem.shape
        
        # Verify shape matches expected
        if actual_shape[0] != expected_seq_len or actual_shape[1] != B or actual_shape[2] != C:
            raise RuntimeError(
                f"memory_attention output shape mismatch: got {actual_shape}, expected "
                f"[seq_len={expected_seq_len}, batch={B}, dim={C}]. "
                f"H={H}, W={W}, H*W={H*W}. "
                f"Total elements: {pix_feat_with_mem.numel()}, Expected: {B * C * H * W}. "
                f"This may indicate an issue with the multimodal encoder fusion."
            )
        
        # Permute from [seq_len, batch, dim] to [batch, dim, seq_len]
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0)
        # Verify shape before view
        if pix_feat_with_mem.shape != (B, C, expected_seq_len):
            raise RuntimeError(
                f"Shape mismatch after permute: got {pix_feat_with_mem.shape}, expected ({B}, {C}, {expected_seq_len})"
            )
        pix_feat_with_mem = pix_feat_with_mem.view(B, C, H, W)
        return pix_feat_with_mem

# ---- 4. _track_step override that passes metas to memory fusion ----------
    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
        view_dirs,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                view_dirs=view_dirs,
                track_in_reverse=track_in_reverse,
            )
            
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    # Override track_step to accept view_dirs for angular memory
    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        view_dirs=None,  # Angular memory: view directions for memory selection
    ):
        """
        Override track_step to accept view_dirs for angular memory selection.
        """
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            view_dirs,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

  
