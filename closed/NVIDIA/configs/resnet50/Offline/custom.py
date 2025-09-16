# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AI_HNC(OfflineGPUBaseConfig):
    system = KnownSystem.AI_HNC

    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 57000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AI_HNC_Triton(AI_HNC):
    use_triton = True
