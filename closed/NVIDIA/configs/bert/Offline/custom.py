# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ACC_AI_HNC(OfflineGPUBaseConfig):
    system = KnownSystem.ACC_AI_HNC

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: dict = {'bert': 1280}

    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    offline_expected_qps = 5700
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ACC_AI_HNC_HighAccuracy(ACC_AI_HNC):
    gpu_batch_size: dict = {'bert': 1024}
    offline_expected_qps = 5000
