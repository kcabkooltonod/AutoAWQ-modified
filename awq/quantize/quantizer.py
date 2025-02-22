import torch
import inspect
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import clear_memory, get_best_device
from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_Marlin,
    WQLinear_GEMVFast,
)
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)


class AwqQuantizer:
    def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        w_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        modules_to_not_convert=None,
        export_compatible=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.export_compatible = export_compatible
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        # 获取到模型、模型的关键参数、第一层的输入数据
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    # def pseudo_quantize_tensor(self, w: torch.Tensor):
    #     org_w_shape = w.shape
    #     if self.group_size > 0:
    #         assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
    #         w = w.reshape(-1, self.group_size)
    #     assert w.dim() == 2
    #     assert torch.isnan(w).sum() == 0

    #     # zero point quantization
    #     if self.zero_point:
    #         max_val = w.amax(dim=1, keepdim=True)
    #         min_val = w.amin(dim=1, keepdim=True)
    #         max_int = 2**self.w_bit - 1
    #         min_int = 0
    #         scales = (max_val - min_val).clamp(min=1e-5) / max_int
    #         zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    #         w = (
    #             torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    #         ) * scales
    #         zeros = zeros.view(org_w_shape[0], -1)
    #     else:
    #         max_val = w.abs().amax(dim=1, keepdim=True)
    #         max_val = max_val.clamp(min=1e-5)
    #         max_int = 2 ** (self.w_bit - 1) - 1
    #         min_int = -(2 ** (self.w_bit - 1))
    #         scales = max_val / max_int
    #         zeros = None
    #         w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    #     assert torch.isnan(scales).sum() == 0
    #     assert torch.isnan(w).sum() == 0

    #     scales = scales.view(org_w_shape[0], -1)
    #     w = w.reshape(org_w_shape)

    #     return w, scales, zeros

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        # print(org_w_shape[0])
        w = w.reshape(org_w_shape[0], -1)
        # print(w.shape)
        group_size_2d = self.group_size  # 二维块的边长
        M, N = w.shape[-2], w.shape[-1]  # 假设权重是二维矩阵
        # print(M, N)
        # 确保输入尺寸能被 group_size_2d 整除（若不能整除，这里会截断）
        blocks_per_row = M // group_size_2d
        blocks_per_col = N // group_size_2d
        cropped_M = blocks_per_row * group_size_2d
        cropped_N = blocks_per_col * group_size_2d

        # 截取可被整除的部分
        w = w[:cropped_M, :cropped_N]

        # 将权重重新组织为二维块结构
        # w: [M, N] -> [blocks_per_row, blocks_per_col, group_size_2d, group_size_2d]
        w = w.view(
            blocks_per_row, group_size_2d,
            blocks_per_col, group_size_2d
        )
        w = w.permute(0, 2, 1, 3)  # [blocks_per_row, blocks_per_col, group_size_2d, group_size_2d]
        w = w.contiguous().view(-1, group_size_2d * group_size_2d)  # [总块数, 块内元素数]

        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True).expand_as(w)
            min_val = w.amin(dim=1, keepdim=True).expand_as(w)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(blocks_per_row, blocks_per_col, group_size_2d, group_size_2d)
            zeros = zeros.permute(0, 2, 1, 3)  # 恢复块的行列顺序
            zeros = zeros.contiguous().view(cropped_M, cropped_N)
            zeros = zeros[:, ::128]
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        # 恢复原始形状（包含截断后的尺寸）
        scales = scales.view(blocks_per_row, blocks_per_col, group_size_2d, group_size_2d)
        scales = scales.permute(0, 2, 1, 3)  # 恢复块的行列顺序
        scales = scales.contiguous().view(cropped_M, cropped_N)
        scales = scales[:, ::128]

        w = w.view(blocks_per_row, blocks_per_col, group_size_2d, group_size_2d)
        w = w.permute(0, 2, 1, 3)  # 恢复块的行列顺序
        w = w.contiguous().view(cropped_M, cropped_N)
        w = w.reshape(org_w_shape)

        # 若原尺寸无法被整除，补零对齐原始形状
        if cropped_M < M or cropped_N < N:
            w_full = torch.zeros(org_w_shape, dtype=w.dtype, device=w.device)
            w_full[:cropped_M, :cropped_N] = w
            w = w_full

            scales_full = torch.zeros(org_w_shape, dtype=scales.dtype, device=w.device)
            scales_full[:cropped_M, :cropped_N] = scales
            scales = scales_full

            if zeros is not None:
                zeros_full = torch.zeros(org_w_shape, dtype=zeros.dtype, device=w.device)
                zeros_full[:cropped_M, :cropped_N] = zeros
                zeros = zeros_full

        return w, scales, zeros

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w

    def quantize(self):
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    # best_device = "cuda:" + str(i % torch.cuda.device_count())
                    best_device = "cuda:2"
                else:
                    best_device = get_best_device()
                print(best_device)
                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)

            # We need to move the rotary embedding every time we move to a new module.
            # Transformers 4.45.0 moved rotary embedding to model definition as of this PR:
            # https://github.com/huggingface/transformers/pull/32617
            self.awq_model.move_embed(self.model, common_device)

            for k, v in self.module_kwargs.items():
                # position embeddings found in tuple
                if isinstance(v, tuple):
                    self.module_kwargs[k] = tuple(
                        item.to(common_device) if isinstance(item, (torch.Tensor, nn.Module)) 
                        else item for item in v
                    )

            # [STEP 1]: Get layer, extract linear modules, extract input features
            # 获取模型的线性网络层，返回一个字典，形式为 [name:nn.Linear()]
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            # 获取并且保存每一层的输入特征，返回一个字典，
            # 形式为 [name:input_features]
            # 同时，将self.inps更新为本transformer层的输出，也就是下一层的输入
            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            # 获取需要scaling的层，返回一个字典的列表
            # 字典中存放了前一层的网络结构，本层的网络结构，输入特征
            # 形式为：(prev_op,layers,inp,...)
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            # 获取每一层的最佳缩放值，返回一个tuple的列表
            # tuple的形式为(prev_op_name, op_name, scales)
            # 一次传入一个字典，也就是一层的信息，比如self_attention或者mlp
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)

            clear_memory()

    def pack(self):
        for i in tqdm(range(len(self.modules)), desc="Packing"):
            named_linears = get_named_linears(self.modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            self._apply_quant(self.modules[i], named_linears)
            clear_memory()

    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(get_best_device()).half()

            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data
            )
            print(linear_layer.weight.data.shape, zeros.shape, scales.shape)

            if self.version == "gemm":
                scales = scales.t().contiguous()
                if zeros is not None:
                    zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM

            elif self.version == "gemv":
                q_linear_module = WQLinear_GEMV

            elif self.version == "marlin":
                q_linear_module = WQLinear_Marlin

            elif self.version == "gemv_fast":
                q_linear_module = WQLinear_GEMVFast

            else:
                raise ValueError(f"Unknown version {self.version}")

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size_2d=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        # 按行把矩阵拼起来
        # weight = torch.cat([_m.weight for _m in layers], dim=0)
        # org_shape = weight.shape
        # print(org_shape)
        # # The weights are reshaped to be organised by quantization group
        # weight = weight.view(-1, self.group_size)
        # # Calculates the relative magnitude of the weights within each of the quantization groups,
        # # and rescales each group individually so that each group has weights on a 0-1 scale.
        # # 对每一行进行绝对值归一化，取绝对值后除最大值
        # w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # # Resizes the rescaled weight matrix back up to its original dimensions
        # w_scale = w_scale.view(org_shape)
        # # Gets the average rescaled magnitude for each output channel
        # w_mean = w_scale.mean(0)
        # clear_memory(weight)

        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape

        # 新增：二维分块逻辑
        #########################################################
        # 确保输入尺寸能被 group_size 整除（若不能整除，这里会截断）
        group_size_2d = self.group_size  # 二维分块的边长
        M, N = org_shape[-2], org_shape[-1]  # 假设权重是二维矩阵

        # 计算可分割的块数并截断
        blocks_per_row = M // group_size_2d
        blocks_per_col = N // group_size_2d
        cropped_M = blocks_per_row * group_size_2d
        cropped_N = blocks_per_col * group_size_2d

        # 截取可被整除的部分
        weight = weight[:cropped_M, :cropped_N]

        # 将权重重新组织为二维块结构
        # [总块数, 块内元素数] = [blocks_per_row*blocks_per_col, group_size_2d^2]
        weight_blocks = weight.view(
            blocks_per_row, group_size_2d,
            blocks_per_col, group_size_2d
        )
        # 调整维度顺序以合并块索引
        weight_blocks = weight_blocks.permute(0, 2, 1, 3)  # [blocks_per_row, blocks_per_col, group_size_2d, group_size_2d]
        weight_blocks = weight_blocks.contiguous().view(-1, group_size_2d * group_size_2d)

        # 按二维块计算归一化
        w_scale = weight_blocks.abs() / (weight_blocks.abs().amax(dim=1, keepdim=True) + 1e-6)

        # 恢复原始形状（包含截断后的尺寸）
        w_scale = w_scale.view(
            blocks_per_row, blocks_per_col,
            group_size_2d, group_size_2d
        )
        w_scale = w_scale.permute(0, 2, 1, 3)  # 恢复块的行列顺序
        w_scale = w_scale.contiguous().view(cropped_M, cropped_N)

        # 若原尺寸无法被整除，补零对齐原始形状
        if cropped_M < M or cropped_N < N:
            w_scale_full = torch.zeros(org_shape, dtype=w_scale.dtype, device=weight.device)
            w_scale_full[:cropped_M, :cropped_N] = w_scale
            w_scale = w_scale_full
        #########################################################

        # 后续计算通道均值（与原逻辑一致）
        w_mean = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)
    # def _compute_best_clip(
    #     self,
    #     w: torch.Tensor,
    #     input_feat: torch.Tensor,
    #     n_grid=20,
    #     max_shrink=0.5,
    #     n_sample_token=512,
    # ):
    #     assert w.dim() == 2
    #     org_w_shape = w.shape  # [co, ci]
    #     group_size_2d = self.group_size  # 二维分块的边长

    #     # 计算可分割的块数并截断
    #     blocks_per_row = org_w_shape[0] // group_size_2d  # 输出通道方向的分块数
    #     blocks_per_col = org_w_shape[1] // group_size_2d  # 输入通道方向的分块数
    #     cropped_co = blocks_per_row * group_size_2d  # 截断后的输出通道数
    #     cropped_ci = blocks_per_col * group_size_2d  # 截断后的输入通道数

    #     # 截取可被整除的部分
    #     w = w[:cropped_co, :cropped_ci]

    #     # 将权重重新组织为二维块结构
    #     # [总块数, 块内元素数] = [blocks_per_row * blocks_per_col, group_size_2d * group_size_2d]
    #     weight_blocks = w.view(
    #         blocks_per_row, group_size_2d,
    #         blocks_per_col, group_size_2d
    #     )
    #     weight_blocks = weight_blocks.permute(0, 2, 1, 3)  # [blocks_per_row, blocks_per_col, group_size_2d, group_size_2d]
    #     weight_blocks = weight_blocks.contiguous().view(-1, group_size_2d * group_size_2d)  # [总块数, 块内元素数]

    #     # 对input_feat进行相同的二维块分组
    #     # input_feat [n_token, ci] -> [1, n_token, n_group, group size*group_size]
    #     input_feat = input_feat.view(-1, input_feat.shape[-1])  # [n_token, ci]
    #     input_feat = input_feat[:, :cropped_ci]  # 截断输入通道
    #     # [1, n_token, blocks_per_col * blocks_per_row, group_size_2d * group_size_2d]
    #     input_feat_blocks = input_feat.view(
    #         1, input_feat.shape[0], blocks_per_col, group_size_2d, blocks_per_row, group_size_2d
    #     )
    #     # [1, n_token, blocks_per_col, blocks_per_row, group_size_2d, group_size_2d]
    #     input_feat_blocks = input_feat_blocks.permute(0, 1, 2, 4, 3, 5)
    #     # [1, n_token, n_group, group_size*group_size]
    #     input_feat_blocks = input_feat_blocks.contiguous().view(1, input_feat.shape[0], -1, group_size_2d * group_size_2d)

    #     # 采样input_feat以减少计算量
    #     step_size = max(1, input_feat_blocks.shape[1] // n_sample_token)
    #     # 
    #     input_feat_blocks_sampled = input_feat_blocks[:, ::step_size]


    #     # 计算每个块的最大值
    #     org_max_val = weight_blocks.abs().amax(dim=-1, keepdim=True).expand_as(weight_blocks)  # [n_group, 1, 1]

    #     # 初始化最佳裁剪值
    #     best_max_val = org_max_val.clone()
    #     min_errs = torch.ones_like(org_max_val) * 1e9

    #     # 计算原始输出
    #     input_feat_blocks_sampled = input_feat_blocks_sampled.to(weight_blocks.device)
    #     org_out = (input_feat_blocks_sampled * weight_blocks).sum(dim=-1)  # [n_group, n_sample_token]

    #     # 遍历裁剪比例，寻找最佳裁剪值
    #     for i_s in range(int(max_shrink * n_grid)):
    #         max_val = org_max_val * (1 - i_s / n_grid)
    #         min_val = -max_val
    #         cur_w = torch.clamp(weight_blocks, min_val, max_val)
    #         q_w = self.pseudo_quantize_tensor(cur_w)[0]
    #         cur_out = (input_feat_blocks_sampled * q_w).sum(dim=-1)

    #         # 计算误差
    #         err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
    #         cur_best_idx = err < min_errs
    #         min_errs[cur_best_idx] = err[cur_best_idx]
    #         best_max_val[cur_best_idx] = max_val[cur_best_idx]

    #     # 恢复最佳裁剪值的形状
    #     best_max_val = best_max_val.view(blocks_per_row, blocks_per_col, group_size_2d, group_size_2d)
    #     best_max_val = best_max_val.permute(0, 2, 1, 3)  # [blocks_per_row, group_size_2d, blocks_per_col, group_size_2d]
    #     best_max_val = best_max_val.contiguous().view(cropped_co, cropped_ci)

    #     # 若原尺寸无法被整除，补零对齐原始形状
    #     if cropped_co < org_w_shape[0] or cropped_ci < org_w_shape[1]:
    #         best_max_val_full = torch.zeros(org_w_shape, dtype=best_max_val.dtype, device=w.device)
    #         best_max_val_full[:cropped_co, :cropped_ci] = best_max_val
    #         best_max_val = best_max_val_full

    #     clear_memory(input_feat_blocks_sampled)
    #     clear_memory(org_out)

    #     return best_max_val


    def init_quant(self, n_samples=128, max_seq_len=512):
        # 获取模型的隐藏层，返回的是nn.ModuleList，
        # 里面的成员是若干个Qwen2DecoderLayer，
        # 再具体地，是一个self attention，一个MLP，两个RMSNorm
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )
        # 返回模型的主体部分（n个transformer block），
        # 第一层的输入（关键字参数和输入数据）
        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if self.awq_model.model_type == "deepseek_v2" or self.awq_model.model_type == "deepseek_v3":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        # 获取所有线性层的输入特征，存放在input_features中
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs
