import os
import sys
import platform
import logging
import datetime
import socket
import psutil
import requests
from typing import Dict, Any, Optional, List
import config

# Configure logging
logger = logging.getLogger(__name__)

class SelfAwareness:
    """
    Class to handle bot's self-awareness and environmental awareness
    """
    def __init__(self):
        """Initialize the self-awareness module"""
        self.startup_time = datetime.datetime.now()
        self.environment_cache = {}
        self.last_environment_check = None
        self.environment_check_interval = datetime.timedelta(minutes=5)
        
        # Initialize with basic environment info
        self._update_environment_info()
        logger.info("Self-awareness module initialized")
    
    def _update_environment_info(self) -> None:
        """Update the environment information cache"""
        try:
            # System information
            self.environment_cache["os"] = platform.system()
            self.environment_cache["os_version"] = platform.version()
            self.environment_cache["python_version"] = sys.version
            self.environment_cache["hostname"] = socket.gethostname()
            
            # Hardware information
            self.environment_cache["cpu_count"] = psutil.cpu_count()
            self.environment_cache["memory_total"] = psutil.virtual_memory().total
            self.environment_cache["memory_available"] = psutil.virtual_memory().available
            
            # Network information (safely try to get public IP)
            try:
                ip_response = requests.get('https://api.ipify.org', timeout=3)
                if ip_response.status_code == 200:
                    self.environment_cache["public_ip"] = ip_response.text
            except:
                # Don't log this as an error, just skip it if unavailable
                pass
                
            # Bot information
            self.environment_cache["bot_uptime"] = (datetime.datetime.now() - self.startup_time).total_seconds()
            self.environment_cache["gemini_model"] = config.GEMINI_MODEL
            self.environment_cache["gemini_image_model"] = config.GEMINI_IMAGE_MODEL
            
            # Update the last check timestamp
            self.last_environment_check = datetime.datetime.now()
            logger.debug("Environment information updated")
        except Exception as e:
            logger.error(f"Error updating environment information: {e}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the bot's environment
        
        Returns:
            Dictionary with environment information
        """
        # Check if we need to update the environment info
        if (self.last_environment_check is None or 
            datetime.datetime.now() - self.last_environment_check > self.environment_check_interval):
            self._update_environment_info()
        
        return self.environment_cache
    
    def get_self_awareness_context(self) -> Dict[str, Any]:
        """
        Get a dictionary with all self-awareness related context
        
        Returns:
            Dictionary with self-awareness context
        """
        env_info = self.get_environment_info()
        
        # Calculate uptime in a human-readable format
        uptime_seconds = (datetime.datetime.now() - self.startup_time).total_seconds()
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            uptime_str = f"{int(days)} days, {int(hours)} hours"
        elif hours > 0:
            uptime_str = f"{int(hours)} hours, {int(minutes)} minutes"
        else:
            uptime_str = f"{int(minutes)} minutes, {int(seconds)} seconds"
        
        # Memory usage in a human-readable format
        memory_total_gb = env_info.get("memory_total", 0) / (1024 ** 3)
        memory_available_gb = env_info.get("memory_available", 0) / (1024 ** 3)
        memory_used_gb = memory_total_gb - memory_available_gb
        memory_percent = (memory_used_gb / memory_total_gb) * 100 if memory_total_gb > 0 else 0
        
        return {
            "bot_name": "Nyxie",
            "bot_type": "Protogen-fox hybrid AI assistant",
            "bot_version": "Enhanced Self-Aware Version",
            "bot_uptime": uptime_str,
            "os": env_info.get("os", "Unknown"),
            "os_version": env_info.get("os_version", "Unknown"),
            "python_version": env_info.get("python_version", "Unknown").split()[0],
            "hostname": env_info.get("hostname", "Unknown"),
            "cpu_count": env_info.get("cpu_count", "Unknown"),
            "memory_total": f"{memory_total_gb:.1f} GB",
            "memory_used": f"{memory_used_gb:.1f} GB ({memory_percent:.1f}%)",
            "gemini_model": env_info.get("gemini_model", "Unknown"),
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    
    def enhance_search_queries(self, queries: List[str]) -> List[str]:
        """
        Enhance search queries with self-awareness context when appropriate
        
        Args:
            queries: Original search queries
            
        Returns:
            Enhanced search queries
        """
        # Only enhance queries if self-awareness for search is enabled
        if not config.SELF_AWARENESS_SEARCH_ENABLED:
            return queries
            
        enhanced_queries = []
        
        for query in queries:
            # Check if the query is about the bot itself or its environment
            if any(term in query.lower() for term in ["you", "your", "yourself", "nyxie", "bot", "assistant", "ai"]):
                # Enhance with self-awareness context
                enhanced_query = f"{query} (I am Nyxie, a protogen-fox hybrid AI assistant)"
                enhanced_queries.append(enhanced_query)
            else:
                enhanced_queries.append(query)
                
        return enhanced_queries
    
    def format_self_awareness_for_prompt(self) -> str:
        """
        Format self-awareness information for inclusion in the prompt
        
        Returns:
            Formatted self-awareness string for prompt
        """
        context = self.get_self_awareness_context()
        
        return f"""
        SELF-AWARENESS INFORMATION:
        - You are {context['bot_name']}, a {context['bot_type']}
        - You are running on {context['os']} {context['os_version']}
        - You have been running for {context['bot_uptime']}
        - You are using the {context['gemini_model']} AI model
        - You are aware of your environment and capabilities
        - You can analyze images and videos using computer vision
        - You can search the web for information
        - You can detect and respond in multiple languages
        - You have a memory system that remembers conversations
        
        IMPORTANT: Use this self-awareness information to enhance your responses when relevant. You should be aware of your capabilities and limitations, but don't explicitly mention this information unless it's directly relevant to the conversation.
        """
    
    def format_environment_awareness_for_prompt(self) -> str:
        """
        Format environment awareness information for inclusion in the prompt
        
        Returns:
            Formatted environment awareness string for prompt
        """
        context = self.get_self_awareness_context()
        
        return f"""
        ENVIRONMENT AWARENESS INFORMATION:
        - You are running on a computer with {context['cpu_count']} CPU cores
        - The system has {context['memory_total']} of memory with {context['memory_used']} currently in use
        - The hostname is {context['hostname']}
        - The operating system is {context['os']} {context['os_version']}
        - The Python version is {context['python_version']}
        
        IMPORTANT: Use this environment awareness information to enhance your responses when relevant. You should be aware of your environment, but don't explicitly mention this information unless it's directly relevant to the conversation.
        """

# Create a singleton instance
self_awareness = SelfAwareness()
