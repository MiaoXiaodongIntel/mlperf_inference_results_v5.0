# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ACC_AI_HNC(OfflineGPUBaseConfig):
    system = KnownSystem.ACC_AI_HNC

    gpu_batch_size = {'gptj': 192}
    offline_expected_qps = 16
    precision = "fp8"
    checkpoint_dir: str = "/work/build/models/GPTJ-6B/checkpoint-final"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ACC_AI_HNC_HighAccuracy(ACC_AI_HNC):
    pass
