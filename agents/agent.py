"""
Evo-Agent: LLM-based Clinical Agent with Evolving Memory

The agent makes predictions about patient status and clinical actions,
then reflects on outcomes to build personalized clinical insights.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from agents.memory import AgentMemory
from agents.memory_manager import ManagedMemory
from model.llms import LLMClient
from prompts.agent_prompt import format_prediction_prompt, format_reflection_prompt
from prompts.memory_management_prompt import format_extraction_prompt, format_consolidation_prompt


class AgentPrediction:
    """Structured prediction from the agent."""

    def __init__(
        self,
        window_index: int,
        vitals_prediction: Dict,
        patient_status_prediction: Dict,
        recommended_actions: List[Dict],
        confidence: Dict,
        raw_response: Dict = None,
    ):
        """
        Initialize agent prediction.

        Args:
            window_index: Index of the window this prediction is for
            vitals_prediction: Predicted vital sign trends
            patient_status_prediction: Predicted patient status and trajectory
            recommended_actions: List of recommended clinical actions
            confidence: Confidence scores and uncertainty factors
            raw_response: Raw LLM response for logging
        """
        self.window_index = window_index
        self.vitals_prediction = vitals_prediction
        self.patient_status_prediction = patient_status_prediction
        self.recommended_actions = recommended_actions
        self.confidence = confidence
        self.raw_response = raw_response
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert prediction to dictionary."""
        return {
            "window_index": self.window_index,
            "timestamp": self.timestamp,
            "vitals_prediction": self.vitals_prediction,
            "patient_status_prediction": self.patient_status_prediction,
            "recommended_actions": self.recommended_actions,
            "confidence": self.confidence,
        }


class EvoAgent:
    """
    Evo-Agent: Clinical decision support agent with evolving memory.

    The agent operates in a continuous learning loop:
    1. Predict patient status and actions for next window
    2. Observe actual events in next window
    3. Reflect on prediction errors
    4. Update memory with new insights
    5. Make next prediction with updated memory
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        log_dir: Optional[str] = None,
        use_managed_memory: bool = False,
        max_memory_entries: int = 5,
    ):
        """
        Initialize Evo-Agent.

        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Model name (defaults to best available)
            api_key: API key (if None, uses environment variable)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            log_dir: Directory to save logs (if None, uses 'logs/agent')
            use_managed_memory: If True, use ManagedMemory instead of AgentMemory
            max_memory_entries: Maximum entries for ManagedMemory (default 5)
        """
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Initialize memory system based on configuration
        self.use_managed_memory = use_managed_memory
        if use_managed_memory:
            self.memory = ManagedMemory(max_entries=max_memory_entries)
        else:
            self.memory = AgentMemory()

        self.predictions: List[AgentPrediction] = []
        self.prediction_count = 0
        self.reflection_count = 0
        self.extraction_count = 0
        self.consolidation_count = 0
        self.total_tokens_used = 0

        # Setup logging
        if log_dir is None:
            log_dir = str(Path(__file__).parent.parent / "logs" / "agent")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Agent logs will be saved to: {self.log_dir}")
        print(f"Memory system: {'Managed' if use_managed_memory else 'Cumulative'}")

    def predict(self, window_data: Dict, window_index: int) -> AgentPrediction:
        """
        Make prediction for the next time window.

        Args:
            window_data: Current window data with history and observations
            window_index: Index of current window

        Returns:
            AgentPrediction object
        """
        # Format memory context
        memory_context = self.memory.format_for_prompt(max_insights=10)

        # Format prompt
        prompt = format_prediction_prompt(window_data, memory_context)

        # Call LLM
        try:
            response = self.llm_client.chat(prompt=prompt, response_format="json")

            # Track usage
            self.prediction_count += 1
            if "usage" in response:
                self.total_tokens_used += response.get("usage", {}).get("input_tokens", 0)
                self.total_tokens_used += response.get("usage", {}).get("output_tokens", 0)

            # Parse response
            if response.get("parsed"):
                parsed = response["parsed"]
            else:
                # Try to extract JSON from content
                content = response["content"]
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(content)

            # Create prediction object
            prediction = AgentPrediction(
                window_index=window_index,
                vitals_prediction=parsed.get("vitals_prediction", {}),
                patient_status_prediction=parsed.get("patient_status_prediction", {}),
                recommended_actions=parsed.get("recommended_actions", []),
                confidence=parsed.get("confidence", {}),
                raw_response=response,
            )

            self.predictions.append(prediction)
            return prediction

        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return default prediction
            return AgentPrediction(
                window_index=window_index,
                vitals_prediction={},
                patient_status_prediction={"severity_score": 0.0, "trajectory": "unknown"},
                recommended_actions=[],
                confidence={"overall_confidence": 0.0},
            )

    def extract_insight(self, current_events: List[Dict], hours_since_admission: float) -> Optional[Dict]:
        """
        Extract ONE clinical insight from current window (for managed memory).

        Args:
            current_events: Events from current 30-minute window
            hours_since_admission: Current time in hours

        Returns:
            Dictionary with system, observation, and status
        """
        if not self.use_managed_memory:
            return None

        # Format extraction prompt
        prompt = format_extraction_prompt(current_events, hours_since_admission)

        # Call LLM
        try:
            response = self.llm_client.chat(prompt=prompt, response_format="json")

            # Track usage
            self.extraction_count += 1
            if "usage" in response:
                self.total_tokens_used += response.get("usage", {}).get("input_tokens", 0)
                self.total_tokens_used += response.get("usage", {}).get("output_tokens", 0)

            # Parse response
            if response.get("parsed"):
                insight = response["parsed"]
            else:
                content = response["content"]
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    insight = json.loads(json_str)
                else:
                    insight = json.loads(content)

            return insight

        except Exception as e:
            print(f"Error during extraction: {e}")
            return None

    def consolidate_memory(self, new_insight: Dict, hours_since_admission: float) -> bool:
        """
        Consolidate new insight into managed memory.

        Args:
            new_insight: Extracted insight (dict with system, observation, status)
            hours_since_admission: Current time in hours

        Returns:
            True if consolidation succeeded, False otherwise
        """
        if not self.use_managed_memory or not new_insight:
            return False

        # Format existing memory
        existing_memory = self.memory.format_for_prompt()

        # Format consolidation prompt
        prompt = format_consolidation_prompt(
            existing_memory=existing_memory,
            new_insight=new_insight,
            hours_since_admission=hours_since_admission,
            max_entries=self.memory.max_entries,
        )

        # Call LLM
        try:
            response = self.llm_client.chat(prompt=prompt, response_format="json")

            # Track usage
            self.consolidation_count += 1
            if "usage" in response:
                self.total_tokens_used += response.get("usage", {}).get("input_tokens", 0)
                self.total_tokens_used += response.get("usage", {}).get("output_tokens", 0)

            # Parse response
            if response.get("parsed"):
                consolidated = response["parsed"]
            else:
                content = response["content"]
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    consolidated = json.loads(json_str)
                else:
                    consolidated = json.loads(content)

            # Update memory with consolidated entries
            clinical_memory = consolidated.get("clinical_memory", [])
            self.memory.clear()

            for entry_data in clinical_memory:
                self.memory.add_or_update_entry(
                    system=entry_data["system"],
                    description=entry_data["description"],
                    last_updated=entry_data["last_updated"],
                    status=entry_data.get("status", "ACTIVE"),
                )

            return True

        except Exception as e:
            print(f"Error during consolidation: {e}")
            return False

    def reflect(self, prediction: AgentPrediction, actual_events: List[Dict]) -> Optional[Dict]:
        """
        Reflect on prediction accuracy and generate new insight.

        Note: This method is primarily for cumulative memory mode.
        In managed memory mode, use extract_insight() and consolidate_memory() instead.

        Args:
            prediction: The agent's previous prediction
            actual_events: Actual events that occurred

        Returns:
            Dictionary containing reflection results and new insight
        """
        # Skip reflection if using managed memory (use extraction/consolidation instead)
        if self.use_managed_memory:
            return None

        # Format memory context
        memory_context = self.memory.format_for_prompt(max_insights=10)

        # Format prompt
        prompt = format_reflection_prompt(
            prediction=prediction.to_dict(),
            actual_events=actual_events,
            window_index=prediction.window_index,
            memory_context=memory_context,
        )

        # Call LLM
        try:
            response = self.llm_client.chat(prompt=prompt, response_format="json")

            # Track usage
            self.reflection_count += 1
            if "usage" in response:
                self.total_tokens_used += response.get("usage", {}).get("input_tokens", 0)
                self.total_tokens_used += response.get("usage", {}).get("output_tokens", 0)

            # Parse response
            if response.get("parsed"):
                reflection = response["parsed"]
            else:
                # Try to extract JSON from content
                content = response["content"]
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    reflection = json.loads(json_str)
                else:
                    reflection = json.loads(content)

            # Add new insight to memory
            new_insight = reflection.get("new_insight", {})
            if new_insight and new_insight.get("insight"):
                self.memory.add_insight(
                    insight=new_insight["insight"],
                    clinical_scenario=new_insight.get("clinical_scenario", "General observation"),
                    source_window=prediction.window_index,
                    confidence=new_insight.get("confidence", 0.5),
                )

            return reflection

        except Exception as e:
            print(f"Error during reflection: {e}")
            return None

    def run_trajectory(self, windows: List[Dict], start_window: int = 0) -> List[Dict]:
        """
        Run agent through a complete patient trajectory.

        Args:
            windows: List of time windows from data parser
            start_window: Which window to start from (default 0)

        Returns:
            List of results for each window
        """
        results = []

        for i in range(start_window, len(windows) - 1):  # -1 because we need next window for reflection
            current_window = windows[i]
            next_window = windows[i + 1]

            print(f"\n{'='*60}")
            print(f"Window {i} (Hour {current_window['hours_since_admission']:.1f})")
            print(f"{'='*60}")

            # Managed Memory Workflow: Extract → Consolidate → Predict
            if self.use_managed_memory:
                # Phase 1: Extract insight from current window
                print(f"Extracting clinical insight...")
                current_events = current_window.get("current_events", [])
                hours = current_window.get("hours_since_admission", 0)

                insight = self.extract_insight(current_events, hours)
                if insight:
                    print(f"  Extracted: {insight.get('system', 'N/A')} - {insight.get('observation', 'N/A')[:60]}...")

                    # Phase 2: Consolidate into memory
                    print(f"Consolidating into memory...")
                    success = self.consolidate_memory(insight, hours)
                    if success:
                        print(f"  Memory updated: {len(self.memory.entries)} entries")

            # Phase 3: Prediction
            print(f"Making prediction for next window...")
            prediction = self.predict(current_window, window_index=i)

            print(f"  Predicted status: {prediction.patient_status_prediction.get('trajectory', 'N/A')}")
            print(f"  Confidence: {prediction.confidence.get('overall_confidence', 0):.2f}")

            # Phase 4: Observation (next window)
            actual_events = next_window.get("current_events", [])
            print(f"  Observed {len(actual_events)} events in next window")

            # Phase 5: Reflection (only for cumulative memory)
            reflection = None
            if not self.use_managed_memory:
                print(f"Reflecting on prediction...")
                reflection = self.reflect(prediction, actual_events)

                if reflection:
                    accuracy = reflection.get("prediction_accuracy", {})
                    print(f"  Prediction accuracy: {accuracy.get('overall_assessment', 'N/A')}")

                    new_insight = reflection.get("new_insight", {})
                    if new_insight.get("insight"):
                        print(f"  New insight: {new_insight['insight'][:80]}...")

            # Store results
            result_entry = {
                "window_index": i,
                "prediction": prediction.to_dict(),
                "actual_events_count": len(actual_events),
                "memory_size": len(self.memory.entries),
            }

            if self.use_managed_memory:
                result_entry["extracted_insight"] = insight
            else:
                result_entry["reflection"] = reflection

            results.append(result_entry)

        print(f"\n{'='*60}")
        print(f"Trajectory complete: {len(results)} windows processed")
        if self.use_managed_memory:
            print(f"Total memory entries: {len(self.memory.entries)}")
            print(f"Active entries: {len(self.memory.get_active_entries())}")
            print(f"Resolved entries: {len(self.memory.get_resolved_entries())}")
        else:
            print(f"Total insights learned: {len(self.memory.entries)}")
        print(f"{'='*60}\n")

        return results

    def save_memory(self, file_path: str):
        """Save agent memory to file."""
        self.memory.save(file_path)

    def load_memory(self, file_path: str):
        """Load agent memory from file."""
        if self.use_managed_memory:
            self.memory = ManagedMemory.load(file_path)
        else:
            self.memory = AgentMemory.load(file_path)

    def get_statistics(self) -> Dict:
        """Get agent usage statistics."""
        stats = {
            "total_predictions": self.prediction_count,
            "total_reflections": self.reflection_count,
            "total_insights": len(self.memory.entries),
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_prediction": (
                self.total_tokens_used / self.prediction_count if self.prediction_count > 0 else 0
            ),
            "memory_type": "managed" if self.use_managed_memory else "cumulative",
        }

        if self.use_managed_memory:
            stats["total_extractions"] = self.extraction_count
            stats["total_consolidations"] = self.consolidation_count
            stats["active_entries"] = len(self.memory.get_active_entries())
            stats["resolved_entries"] = len(self.memory.get_resolved_entries())

        return stats
