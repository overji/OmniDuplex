"""Causal text inputs from recorded Agent tool results, never reference answers."""
from benchmark.data.timestamps import seconds


AGENT_INPUT_PROTOCOL = "timestamped_agent_results_and_post_media_generation_v2"


def agent_results(sample):
    if sample.get("mode") != "agent":
        return []
    results = []
    for index, response in enumerate(sample.get("responses", [])):
        messages = response.get("messages", [response])
        for message in messages:
            if message.get("role") != "tool":
                continue
            timestamp = message.get("timestamp", message.get("trigger_time"))
            content = message.get("content", message.get("response"))
            at = seconds(timestamp)
            if at < seconds(sample["question_time"]) or not isinstance(content, str):
                raise ValueError("Invalid Agent result timestamp or text")
            results.append({"timestamp": timestamp, "at_seconds": at,
                            "text": content, "source_response_index": index})
    return sorted(results, key=lambda result: result["at_seconds"])


class AgentInputs:
    def __init__(self, sample):
        self.pending = list(sample.get("agent_results", []))
        self.delivered = []

    def take(self, clock, *, inclusive=True):
        texts = []
        while self.pending and (self.pending[0]["at_seconds"] <= clock if inclusive
                               else self.pending[0]["at_seconds"] < clock):
            result = self.pending.pop(0)
            self.delivered.append({**result, "input_clock_seconds": clock})
            texts.append(f"Agent returned a recorded simulated result at video time "
                         f"{result['timestamp']}:\n{result['text']}\n"
                         "Report this result to the user after this time. Do not invent "
                         "additional actions or treat this recorded simulation as a real execution.")
        return texts
