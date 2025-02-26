class LifecycleManager:
    """Manages component lifecycles and system state transitions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.state_machine = ComponentStateMachine()
        self.health_monitor = ComponentHealthMonitor(config)
        self.recovery_handler = RecoveryHandler(config)
        self.components = {}
        self.dependencies = defaultdict(set)
        self.status_history = []
        
    async def start_component(self, name: str):
        """Start a registered component"""
        try:
            if name not in self.components:
                raise ValueError(f"Component {name} not registered")
                
            component = self.components[name]
            
            # Check dependencies
            if not await self._check_dependencies(name):
                raise RuntimeError(f"Dependencies not ready for {name}")
                
            # Transition to starting state
            await self.state_machine.transition(name, "starting")
            
            # Initialize component
            if hasattr(component, "initialize"):
                await component.initialize()
                
            # Start component
            if hasattr(component, "start"):
                await component.start()
                
            # Update state
            await self.state_machine.transition(name, "running")
            
            # Start health monitoring
            await self.health_monitor.monitor_component(name, component)
            
            logging.info(f"Component {name} started successfully")
            
        except Exception as e:
            logging.error(f"Error starting component {name}: {e}")
            await self.handle_component_error(name, e)
            raise
            
    async def stop_component(self, name: str):
        """Stop a running component"""
        try:
            if name not in self.components:
                raise ValueError(f"Component {name} not registered")
                
            component = self.components[name]
            
            # Check dependent components
            dependents = self._get_dependents(name)
            if dependents:
                for dependent in dependents:
                    await self.stop_component(dependent)
                    
            # Transition to stopping state
            await self.state_machine.transition(name, "stopping")
            
            # Stop component
            if hasattr(component, "stop"):
                await component.stop()
                
            # Cleanup
            if hasattr(component, "cleanup"):
                await component.cleanup()
                
            # Update state
            await self.state_machine.transition(name, "stopped")
            
            # Stop health monitoring
            await self.health_monitor.stop_monitoring(name)
            
            logging.info(f"Component {name} stopped successfully")
            
        except Exception as e:
            logging.error(f"Error stopping component {name}: {e}")
            await self.handle_component_error(name, e)
            raise
            
    async def register_component(self, name: str, component: Any, 
                               dependencies: List[str] = None):
        """Register a new component"""
        if name in self.components:
            raise ValueError(f"Component {name} already registered")
            
        self.components[name] = component
        
        if dependencies:
            self.dependencies[name].update(dependencies)
            
        # Initialize state
        await self.state_machine.add_component(name)
        
        logging.info(f"Component {name} registered with dependencies: {dependencies}")
        
    async def handle_component_error(self, name: str, error: Exception):
        """Handle component error"""
        try:
            # Log error
            logging.error(f"Component {name} error: {error}")
            
            # Update state
            await self.state_machine.transition(name, "error")
            
            # Add to status history
            self.status_history.append({
                "component": name,
                "type": "error",
                "error": str(error),
                "timestamp": time.time()
            })
            
            # Attempt recovery
            await self.recovery_handler.handle_error(name, error)
            
        except Exception as e:
            logging.error(f"Error handling component error: {e}")
            
    async def _check_dependencies(self, name: str) -> bool:
        """Check if component dependencies are ready"""
        dependencies = self.dependencies[name]
        
        for dep in dependencies:
            state = await self.state_machine.get_state(dep)
            if state != "running":
                return False
                
        return True
        
    def _get_dependents(self, name: str) -> Set[str]:
        """Get components that depend on given component"""
        dependents = set()
        
        for component, deps in self.dependencies.items():
            if name in deps:
                dependents.add(component)
                
        return dependents
        
class ComponentStateMachine:
    """Manages component state transitions"""
    
    def __init__(self):
        self.states = {}
        self.valid_transitions = {
            "registered": ["starting"],
            "starting": ["running", "error"],
            "running": ["stopping", "error"],
            "stopping": ["stopped", "error"],
            "stopped": ["starting"],
            "error": ["starting", "stopped"]
        }
        
    async def add_component(self, name: str):
        """Add component to state machine"""
        self.states[name] = "registered"
        
    async def transition(self, name: str, new_state: str):
        """Transition component to new state"""
        if name not in self.states:
            raise ValueError(f"Component {name} not in state machine")
            
        current_state = self.states[name]
        
        if new_state not in self.valid_transitions[current_state]:
            raise ValueError(
                f"Invalid transition {current_state} -> {new_state}"
            )
            
        self.states[name] = new_state
        
    async def get_state(self, name: str) -> str:
        """Get current state of component"""
        if name not in self.states:
            raise ValueError(f"Component {name} not in state machine")
            
        return self.states[name]

class ComponentHealthMonitor:
    """Monitors component health"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.health_checks = {}
        self.health_history = defaultdict(list)
        self.running = True
        self.monitor_task = None
        
    async def monitor_component(self, name: str, component: Any):
        """Start monitoring component"""
        self.health_checks[name] = {
            "component": component,
            "last_check": time.time(),
            "health": 1.0
        }
        
        # Start monitoring if not already running
        if not self.monitor_task:
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            
    async def stop_monitoring(self, name: str):
        """Stop monitoring component"""
        if name in self.health_checks:
            del self.health_checks[name]
            
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                for name, check in self.health_checks.items():
                    await self._check_component_health(name, check)
                    
                await asyncio.sleep(
                    self.config.get("health_check_interval", 5)
                )
                
            except Exception as e:
                logging.error(f"Health monitor error: {e}")
                
    async def _check_component_health(self, name: str, check: Dict):
        """Check health of individual component"""
        try:
            component = check["component"]
            
            # Get component health
            health = await self._get_component_health(component)
            
            # Update health status
            check["health"] = health
            check["last_check"] = time.time()
            
            # Add to history
            self.health_history[name].append({
                "health": health,
                "timestamp": time.time()
            })
            
            # Trim history
            max_history = self.config.get("max_health_history", 1000)
            if len(self.health_history[name]) > max_history:
                self.health_history[name] = self.health_history[name][-max_history:]
                
            # Check for degradation
            if health < self.config.get("min_health", 0.5):
                await self._handle_health_degradation(name, health)
                
        except Exception as e:
            logging.error(f"Health check error for {name}: {e}")
            
    async def _get_component_health(self, component: Any) -> float:
        """Get health score for component"""
        try:
            if hasattr(component, "get_health"):
                return await component.get_health()
                
            if hasattr(component, "health"):
                return float(component.health)
                
            # Default health check
            return self._check_basic_health(component)
            
        except Exception:
            return 0.0
            
    def _check_basic_health(self, component: Any) -> float:
        """Basic component health check"""
        # Check if component is responsive
        try:
            # Check attributes exist
            if not hasattr(component, "__dict__"):
                return 0.0
                
            # Check memory usage
            import sys
            mem_size = sys.getsizeof(component)
            if mem_size > self.config.get("max_component_size", 1e9):
                return 0.0
                
            return 1.0
            
        except Exception:
            return 0.0
            
    async def _handle_health_degradation(self, name: str, health: float):
        """Handle component health degradation"""
        logging.warning(
            f"Component {name} health degraded to {health}"
        )
        
        # Notify recovery handler
        await self.recovery_handler.handle_health_degradation(
            name, health
        )

class RecoveryHandler:
    """Handles component recovery"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.recovery_history = defaultdict(list)
        
    async def handle_error(self, name: str, error: Exception):
        """Handle component error"""
        try:
            # Add to recovery history
            self.recovery_history[name].append({
                "type": "error",
                "error": str(error),
                "timestamp": time.time()
            })
            
            # Check error frequency
            if self._check_error_frequency(name) > self.config.get("max_error_frequency", 0.1):
                await self._handle_frequent_errors(name)
                return
                
            # Attempt recovery
            await self._attempt_recovery(name)
            
        except Exception as e:
            logging.error(f"Recovery error for {name}: {e}")
            
    async def handle_health_degradation(self, name: str, health: float):
        """Handle component health degradation"""
        try:
            # Add to recovery history
            self.recovery_history[name].append({
                "type": "degradation",
                "health": health,
                "timestamp": time.time()
            })
            
            # Check degradation frequency
            if self._check_degradation_frequency(name) > self.config.get("max_degradation_frequency", 0.1):
                await self._handle_frequent_degradation(name)
                return
                
            # Attempt recovery
            await self._attempt_recovery(name)
            
        except Exception as e:
            logging.error(f"Recovery error for {name}: {e}")
            
    async def _attempt_recovery(self, name: str):
        """Attempt component recovery"""
        try:
            component = self.components[name]
            
            # Stop component
            await self.stop_component(name)
            
            # Cleanup
            if hasattr(component, "cleanup"):
                await component.cleanup()
                
            # Reset state
            if hasattr(component, "reset"):
                await component.reset()
                
            # Restart component
            await self.start_component(name)
            
            logging.info(f"Component {name} recovered successfully")
            
        except Exception as e:
            logging.error(f"Recovery failed for {name}: {e}")
            
    async def _handle_frequent_errors(self, name: str):
        """Handle frequent component errors"""
        logging.error(f"Frequent errors detected for {name}")
        
        # Stop component
        await self.stop_component(name)
        
        # Notify administrators
        await self._notify_admins({
            "component": name,
            "type": "frequent_errors",
            "history": self.recovery_history[name][-10:]
        })
        
    async def _handle_frequent_degradation(self, name: str):
        """Handle frequent component health degradation"""
        logging.error(f"Frequent health degradation for {name}")
        
        # Stop component
        await self.stop_component(name)
        
        # Notify administrators
        await self._notify_admins({
            "component": name,
            "type": "frequent_degradation",
            "history": self.recovery_history[name][-10:]
        })
        
    def _check_error_frequency(self, name: str) -> float:
        """Check component error frequency"""
        recent_errors = [
            r for r in self.recovery_history[name]
            if r["type"] == "error" and
            time.time() - r["timestamp"] < 3600
        ]
        
        return len(recent_errors) / 3600
        
    def _check_degradation_frequency(self, name: str) -> float:
        """Check component degradation frequency"""
        recent_degradations = [
            r for r in self.recovery_history[name]
            if r["type"] == "degradation" and
            time.time() - r["timestamp"] < 3600
        ]
        
        return len(recent_degradations) / 3600

