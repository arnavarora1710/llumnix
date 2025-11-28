import time
import asyncio
from typing import Dict, Tuple, List, Optional, Callable, Any
from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d

import ray
import ray.actor

from llumnix.backends.backend_interface import BackendMigrationInterface
from llumnix.ray_utils import (
    LlumnixActor,
    list_actor_names_by_actor_type,
    get_llumnix_actor_id,
    get_llumnix_actor_handle,
)
from llumnix.logging.logger import init_logger
from llumnix.utils import MigrationResponse

logger = init_logger(__name__)


class GPUP2PProfiler:
    """
    Profiles GPU P2P communication between Ray actor nodes by sending
    incrementally larger KV cache blocks and constructing interpolation
    functions to estimate transfer times.
    """
    
    def __init__(self, backend_engine: BackendMigrationInterface):
        self.backend_engine = backend_engine
        self.profiling_data: Dict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
        self.interpolation_functions: Dict[Tuple[str, str], Callable[[int], float]] = {}
        self.node_ids: List[str] = []
        self.instance_actors: Dict[str, ray.actor.ActorHandle] = {}
        self._profiling_complete = False
        
    async def _get_all_instance_actors(self) -> Tuple[List[str], Dict[str, ray.actor.ActorHandle]]:
        """
        Get all instance actors (nodes) in the cluster.
        
        Returns:
            Tuple of (node_ids list, dict mapping node_id to actor handle)
        """
        actor_names = list_actor_names_by_actor_type(LlumnixActor.INSTANCE)
        node_ids = []
        instance_actors = {}
        
        for actor_name in actor_names:
            try:
                actor_handle = ray.get_actor(actor_name, namespace="llumnix")
                node_id = get_llumnix_actor_id(LlumnixActor.INSTANCE, actor_name)
                node_ids.append(node_id)
                instance_actors[node_id] = actor_handle
                logger.info(f"Found instance actor: {node_id}")
            except Exception as e:
                logger.warning(f"Failed to get actor {actor_name}: {e}")
        
        return node_ids, instance_actors
    
    async def _profile_pair(
        self,
        src_node_id: str,
        dst_node_id: str,
        src_actor: ray.actor.ActorHandle,
        dst_actor: ray.actor.ActorHandle,
        block_sizes: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Profile transfer time between a pair of nodes for different block sizes.
        
        Args:
            src_node_id: Source node identifier
            dst_node_id: Destination node identifier
            src_actor: Source instance actor handle
            dst_actor: Destination instance actor handle
            block_sizes: List of block counts to profile
            
        Returns:
            List of (block_count, transfer_time) tuples
        """
        results = []
        
        # Use a dummy request ID for profiling
        dummy_request_id = f"_profiling_{src_node_id}_{dst_node_id}"
        
        logger.info(f"Profiling {src_node_id} -> {dst_node_id} with block sizes: {block_sizes}")
        
        for num_blocks in block_sizes:
            try:
                # Prepare source and destination block indices
                # Use sequential block indices starting from 0
                src_blocks = list(range(num_blocks))
                dst_blocks = list(range(num_blocks))
                
                # Measure transfer time
                start_time = time.perf_counter()
                
                # Send cache from source to destination
                response: MigrationResponse = await self.backend_engine.send_cache(
                    dst_instance_actor=dst_actor,
                    src_blocks=src_blocks,
                    dst_blocks=dst_blocks,
                    request_id=dummy_request_id,
                    is_last_stage=False,
                )
                
                end_time = time.perf_counter()
                transfer_time = end_time - start_time
                
                if response.success:
                    results.append((num_blocks, transfer_time))
                    logger.debug(
                        f"  {num_blocks} blocks: {transfer_time:.4f}s "
                        f"({num_blocks/transfer_time:.2f} blocks/s)"
                    )
                else:
                    logger.warning(
                        f"  Failed to transfer {num_blocks} blocks from {src_node_id} to {dst_node_id}"
                    )
                    
            except Exception as e:
                logger.error(
                    f"Error profiling {num_blocks} blocks from {src_node_id} to {dst_node_id}: {e}",
                    exc_info=True
                )
                # Continue with next block size even if one fails
                continue
        
        return results
    
    async def run_profiling(
        self,
        min_blocks: int = 1,
        max_blocks: int = 100,
        num_samples: int = 10,
        warmup_blocks: int = 1,
    ) -> None:
        """
        Run the initial profiling phase that sends incrementally larger KV cache
        blocks between all pairs of nodes.
        
        Args:
            min_blocks: Minimum number of blocks to profile
            max_blocks: Maximum number of blocks to profile
            num_samples: Number of different block sizes to test
            warmup_blocks: Number of blocks for warmup transfer (to initialize connections)
        """
        logger.info("Starting GPU P2P profiling phase")
        
        # Get all instance actors
        node_ids, instance_actors = await self._get_all_instance_actors()
        
        if len(node_ids) < 2:
            logger.warning(
                f"Only {len(node_ids)} instance(s) found. Need at least 2 for P2P profiling."
            )
            return
        
        self.node_ids = node_ids
        self.instance_actors = instance_actors
        
        # Generate block sizes to profile (logarithmically spaced)
        block_sizes = np.logspace(
            np.log10(max(1, min_blocks)),
            np.log10(max_blocks),
            num=num_samples,
            dtype=int
        )
        # Remove duplicates and sort
        block_sizes = sorted(list(set(block_sizes)))
        
        logger.info(f"Profiling {len(node_ids)} nodes with block sizes: {block_sizes}")
        
        # Warmup: do a small transfer to initialize connections
        if warmup_blocks > 0:
            logger.info("Running warmup transfers...")
            for src_node_id, src_actor in instance_actors.items():
                for dst_node_id, dst_actor in instance_actors.items():
                    if src_node_id != dst_node_id:
                        try:
                            await self._profile_pair(
                                src_node_id, dst_node_id, src_actor, dst_actor, [warmup_blocks]
                            )
                        except Exception as e:
                            logger.warning(f"Warmup failed for {src_node_id} -> {dst_node_id}: {e}")
        
        # Profile all pairs
        tasks = []
        for src_node_id, src_actor in instance_actors.items():
            for dst_node_id, dst_actor in instance_actors.items():
                if src_node_id != dst_node_id:
                    task = self._profile_pair(
                        src_node_id, dst_node_id, src_actor, dst_actor, block_sizes
                    )
                    tasks.append((src_node_id, dst_node_id, task))
        
        # Run profiling tasks
        for src_node_id, dst_node_id, task in tasks:
            try:
                results = await task
                if results:
                    self.profiling_data[(src_node_id, dst_node_id)] = results
                    logger.info(
                        f"Completed profiling {src_node_id} -> {dst_node_id}: "
                        f"{len(results)} successful measurements"
                    )
            except Exception as e:
                logger.error(
                    f"Error profiling {src_node_id} -> {dst_node_id}: {e}",
                    exc_info=True
                )
        
        # Build interpolation functions
        self._build_interpolation_functions()
        
        self._profiling_complete = True
        logger.info("GPU P2P profiling phase completed")
    
    def _build_interpolation_functions(self) -> None:
        """
        Build interpolation functions for each node pair based on profiling data.
        """
        for (src_node_id, dst_node_id), data_points in self.profiling_data.items():
            if len(data_points) < 2:
                logger.warning(
                    f"Insufficient data points for {src_node_id} -> {dst_node_id}, "
                    f"skipping interpolation"
                )
                continue
            
            # Extract block counts and transfer times
            block_counts = np.array([x[0] for x in data_points])
            transfer_times = np.array([x[1] for x in data_points])
            
            # Sort by block count
            sort_idx = np.argsort(block_counts)
            block_counts = block_counts[sort_idx]
            transfer_times = transfer_times[sort_idx]
            
            # Create interpolation function
            # Use linear interpolation with extrapolation
            try:
                interp_func = interp1d(
                    block_counts,
                    transfer_times,
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                self.interpolation_functions[(src_node_id, dst_node_id)] = interp_func
                logger.debug(
                    f"Built interpolation function for {src_node_id} -> {dst_node_id} "
                    f"with {len(block_counts)} data points"
                )
            except Exception as e:
                logger.error(
                    f"Failed to build interpolation for {src_node_id} -> {dst_node_id}: {e}"
                )
    
    def estimate_transfer_time(
        self,
        src_node_id: str,
        dst_node_id: str,
        msg_size_blocks: int,
    ) -> Optional[float]:
        """
        Estimate the time to send msg_size_blocks amount of data between nodes.
        
        Args:
            src_node_id: Source node identifier
            dst_node_id: Destination node identifier
            msg_size_blocks: Number of blocks to transfer
            
        Returns:
            Estimated transfer time in seconds, or None if no data available
        """
        if not self._profiling_complete:
            logger.warning("Profiling not completed yet. Run run_profiling() first.")
            return None
        
        key = (src_node_id, dst_node_id)
        
        if key not in self.interpolation_functions:
            logger.warning(
                f"No profiling data available for {src_node_id} -> {dst_node_id}"
            )
            return None
        
        try:
            interp_func = self.interpolation_functions[key]
            estimated_time = float(interp_func(msg_size_blocks))
            # Ensure non-negative time
            return max(0.0, estimated_time)
        except Exception as e:
            logger.error(
                f"Error estimating transfer time for {src_node_id} -> {dst_node_id}: {e}"
            )
            return None
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the profiling results.
        
        Returns:
            Dictionary containing profiling summary information
        """
        summary = {
            "num_nodes": len(self.node_ids),
            "num_pairs_profiled": len(self.profiling_data),
            "profiling_complete": self._profiling_complete,
        }
        
        if self.profiling_data:
            # Calculate statistics
            all_times = []
            for data_points in self.profiling_data.values():
                all_times.extend([x[1] for x in data_points])
            
            if all_times:
                summary["min_transfer_time"] = min(all_times)
                summary["max_transfer_time"] = max(all_times)
                summary["avg_transfer_time"] = np.mean(all_times)
        
        return summary
