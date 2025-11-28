# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo

logger = init_logger(__name__)


class MigrationFilterConfig:
    def __init__(self, migrate_out_load_threshold, gpu_p2p_profiler=None, max_transfer_time_threshold=None):
        self.migrate_out_load_threshold: float = migrate_out_load_threshold
        self.gpu_p2p_profiler = gpu_p2p_profiler  # Optional GPUP2PProfiler instance
        self.max_transfer_time_threshold: Optional[float] = max_transfer_time_threshold  # Max transfer time in seconds


class MigrationFilter(ABC):
    @abstractmethod
    def filter_src_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        raise NotImplementedError

    @abstractmethod
    def filter_dst_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        raise NotImplementedError


class MigrationFilterPipeline:
    def __init__(self, filter_config: MigrationFilterConfig) -> None:
        self.filter_config = filter_config
        self.registered_filters: Dict[str, MigrationFilter] = {}

    def __repr__(self):
        return "MigrationFilterPipeline(filters={})".format(self.registered_filters.keys())

    def add_filter(self, filter_name: str, migration_filter: MigrationFilter) -> bool:
        if filter_name in self.registered_filters:
            logger.warning("Migration filter {} has been registered.".format(filter_name))
            return False

        self.registered_filters[filter_name] = migration_filter
        return True

    def remove_filter(self, filter_name: str) -> None:
        self.registered_filters.pop(filter_name, None)

    def get_filter(self, filter_name: str) -> Optional[MigrationFilter]:
        return self.registered_filters.get(filter_name, None)

    def filter_instances(self, instance_infos: List[InstanceInfo]) -> Tuple[List[InstanceInfo], List[InstanceInfo]]:
        src_filter_conditions = [filter.filter_src_condition(self.filter_config)
                                 for filter in self.registered_filters.values()]
        dst_filter_conditions = [filter.filter_dst_condition(self.filter_config)
                                 for filter in self.registered_filters.values()]
        filtered_src_instance_infos = [info for info in instance_infos if all(cond(info) for cond in src_filter_conditions)]
        filtered_dst_instance_infos = [info for info in instance_infos if all(cond(info) for cond in dst_filter_conditions)]
        return filtered_src_instance_infos, filtered_dst_instance_infos

    def filter_src_instances(self, instance_infos: List[InstanceInfo]) -> List[InstanceInfo]:
        src_filter_conditions = [filter.filter_src_condition(self.filter_config)
                                 for filter in self.registered_filters.values()]
        filtered_src_instance_infos = [info for info in instance_infos if all(cond(info) for cond in src_filter_conditions)]
        return filtered_src_instance_infos

    def filter_dst_instances(self, instance_infos: List[InstanceInfo]) -> List[InstanceInfo]:
        dst_filter_conditions = [filter.filter_dst_condition(self.filter_config)
                                 for filter in self.registered_filters.values()]
        filtered_dst_instance_infos = [info for info in instance_infos if all(cond(info) for cond in dst_filter_conditions)]
        return filtered_dst_instance_infos


class LoadFilter(MigrationFilter):
    def filter_src_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        def compare_load(instance_info: InstanceInfo) -> bool:
            metrics_cls = type(instance_info.migration_load_metric)
            migrate_out_load_threshold = metrics_cls(filter_config.migrate_out_load_threshold)
            return instance_info.num_killed_requests > 0 \
                or migrate_out_load_threshold < instance_info.migration_load_metric
        return compare_load

    def filter_dst_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        def compare_load(instance_info: InstanceInfo) -> bool:
            metrics_cls = type(instance_info.migration_load_metric)
            migrate_out_load_threshold = metrics_cls(filter_config.migrate_out_load_threshold)
            return instance_info.num_killed_requests == 0 \
                and instance_info.migration_load_metric < migrate_out_load_threshold

        return compare_load


class CustomFilter(MigrationFilter):
    def __init__(self):
        super().__init__()
        self.src_filter = lambda _: True
        self.dst_filter = lambda _: True

    def set_filter_condtition(self, src_filter: Optional[Callable[[InstanceInfo], bool]] = None,
                              dst_filter: Optional[Callable[[InstanceInfo], bool]] = None) -> None:
        if src_filter:
            self.src_filter = src_filter
        if dst_filter:
            self.dst_filter = dst_filter

    def filter_src_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        return self.src_filter

    def filter_dst_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        return self.dst_filter


class LoadAndProfilingFilter(MigrationFilter):
    """
    Filter that considers both the out load (migration_load_metric_after_migrate_out)
    and profiling data (estimated transfer time from GPU P2P profiler).
    """
    def filter_src_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        def compare_load_and_profiling(instance_info: InstanceInfo) -> bool:
            # Check out load threshold
            metrics_cls = type(instance_info.migration_load_metric)
            migrate_out_load_threshold = metrics_cls(filter_config.migrate_out_load_threshold)
            
            # Source should have high load or killed requests
            has_high_load = instance_info.num_killed_requests > 0 \
                or migrate_out_load_threshold < instance_info.migration_load_metric
            
            # Check that out load after migration is acceptable
            out_load_acceptable = True
            if hasattr(instance_info, 'migration_load_metric_after_migrate_out'):
                out_load_metric = instance_info.migration_load_metric_after_migrate_out
                # Out load should be lower than current load (migration should help)
                out_load_acceptable = out_load_metric < instance_info.migration_load_metric
            
            return has_high_load and out_load_acceptable
        
        return compare_load_and_profiling

    def filter_dst_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        def compare_load_and_profiling(instance_info: InstanceInfo) -> bool:
            # Check load threshold
            metrics_cls = type(instance_info.migration_load_metric)
            migrate_out_load_threshold = metrics_cls(filter_config.migrate_out_load_threshold)
            
            # Destination should have low load and no killed requests
            has_low_load = instance_info.num_killed_requests == 0 \
                and instance_info.migration_load_metric < migrate_out_load_threshold
            
            # If profiling is enabled, check that we can estimate transfer time
            # (profiling data should be available)
            profiling_ok = True
            if filter_config.gpu_p2p_profiler is not None:
                # Check if profiler has completed profiling
                if hasattr(filter_config.gpu_p2p_profiler, '_profiling_complete'):
                    profiling_ok = filter_config.gpu_p2p_profiler._profiling_complete
                else:
                    profiling_ok = False
            
            return has_low_load and profiling_ok
        
        return compare_load_and_profiling


class LoadAndProfilingPairFilter(MigrationFilter):
    """
    Filter that considers both out load and profiling data when evaluating migration pairs.
    This filter needs to be used with a custom approach since it needs both src and dst info.
    """
    def __init__(self):
        super().__init__()
        self._src_instance_infos: List[InstanceInfo] = []
        self._dst_instance_infos: List[InstanceInfo] = []
    
    def set_instance_infos(self, src_instance_infos: List[InstanceInfo], 
                          dst_instance_infos: List[InstanceInfo]) -> None:
        """Set the instance infos to enable pair-based filtering."""
        self._src_instance_infos = src_instance_infos
        self._dst_instance_infos = dst_instance_infos
    
    def filter_src_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        def compare_with_profiling(instance_info: InstanceInfo) -> bool:
            # Check out load threshold
            metrics_cls = type(instance_info.migration_load_metric)
            migrate_out_load_threshold = metrics_cls(filter_config.migrate_out_load_threshold)
            
            # Source should have high load or killed requests
            has_high_load = instance_info.num_killed_requests > 0 \
                or migrate_out_load_threshold < instance_info.migration_load_metric
            
            # Check out load after migration
            out_load_acceptable = True
            if hasattr(instance_info, 'migration_load_metric_after_migrate_out'):
                out_load_metric = instance_info.migration_load_metric_after_migrate_out
                out_load_acceptable = out_load_metric < instance_info.migration_load_metric
            
            # If profiling is available, check if there's at least one destination
            # with acceptable transfer time
            if filter_config.gpu_p2p_profiler is not None and \
               hasattr(filter_config.gpu_p2p_profiler, '_profiling_complete') and \
               filter_config.gpu_p2p_profiler._profiling_complete:
                
                # Check if there's at least one destination with acceptable transfer time
                has_acceptable_dst = False
                num_blocks = getattr(instance_info, 'num_blocks_last_running_request', 0)
                
                for dst_info in self._dst_instance_infos:
                    if dst_info.instance_id == instance_info.instance_id:
                        continue
                    
                    # Estimate transfer time
                    estimated_time = filter_config.gpu_p2p_profiler.estimate_transfer_time(
                        src_node_id=instance_info.instance_id,
                        dst_node_id=dst_info.instance_id,
                        msg_size_blocks=num_blocks
                    )
                    
                    # Check if transfer time is acceptable
                    if estimated_time is not None:
                        if filter_config.max_transfer_time_threshold is None or \
                           estimated_time <= filter_config.max_transfer_time_threshold:
                            has_acceptable_dst = True
                            break
                    else:
                        # If no profiling data, allow migration (fallback)
                        has_acceptable_dst = True
                        break
                
                return has_high_load and out_load_acceptable and has_acceptable_dst
            
            return has_high_load and out_load_acceptable
        
        return compare_with_profiling

    def filter_dst_condition(self, filter_config: MigrationFilterConfig) -> Callable[[InstanceInfo], bool]:
        def compare_with_profiling(instance_info: InstanceInfo) -> bool:
            # Check load threshold
            metrics_cls = type(instance_info.migration_load_metric)
            migrate_out_load_threshold = metrics_cls(filter_config.migrate_out_load_threshold)
            
            # Destination should have low load and no killed requests
            has_low_load = instance_info.num_killed_requests == 0 \
                and instance_info.migration_load_metric < migrate_out_load_threshold
            
            # If profiling is available, check if there's at least one source
            # with acceptable transfer time
            if filter_config.gpu_p2p_profiler is not None and \
               hasattr(filter_config.gpu_p2p_profiler, '_profiling_complete') and \
               filter_config.gpu_p2p_profiler._profiling_complete:
                
                # Check if there's at least one source with acceptable transfer time
                has_acceptable_src = False
                
                for src_info in self._src_instance_infos:
                    if src_info.instance_id == instance_info.instance_id:
                        continue
                    
                    num_blocks = getattr(src_info, 'num_blocks_last_running_request', 0)
                    
                    # Estimate transfer time
                    estimated_time = filter_config.gpu_p2p_profiler.estimate_transfer_time(
                        src_node_id=src_info.instance_id,
                        dst_node_id=instance_info.instance_id,
                        msg_size_blocks=num_blocks
                    )
                    
                    # Check if transfer time is acceptable
                    if estimated_time is not None:
                        if filter_config.max_transfer_time_threshold is None or \
                           estimated_time <= filter_config.max_transfer_time_threshold:
                            has_acceptable_src = True
                            break
                    else:
                        # If no profiling data, allow migration (fallback)
                        has_acceptable_src = True
                        break
                
                return has_low_load and has_acceptable_src
            
            return has_low_load
        
        return compare_with_profiling


class MigrationFilterFactory:
    _POLICY_REGISTRY = {
        'load': LoadFilter,
        'custom': CustomFilter,
        'load_and_profiling': LoadAndProfilingFilter,
        'load_and_profiling_pair': LoadAndProfilingPairFilter,
    }

    @classmethod
    def get_filter(cls, filter_name: str, **kwargs) -> MigrationFilter:
        return cls._POLICY_REGISTRY[filter_name](**kwargs)
