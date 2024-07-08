"""Extends vanilla strategy for cursor replay strategy that allows the model to paint a red dot on the
   suggested target location and optionally self-corrects.

    Sends a series of screenshots to the language model, asks for a description
    of what happened, and then provides a sequence of actions in concrete coordinates and keyboard
    inputs along with proposed modifications in natural language. It then outputs the new action sequence.

    The strategy incorporates visual feedback by painting a red dot on the screenshot and 
    enables the model to confirm or correct the suggested actions based on this feedback.
    ...

1. Given the recorded states, describe what happened
2. Given the description of what happened, proposed modifications in natural language
instructions, the current state, and the actions produced so far, produce the next
action.
"""

from loguru import logger

from PIL import Image, ImageDraw

import io

import base64

from openadapt.strategies.vanilla import VanillaReplayStrategy

from openadapt import adapters, models, strategies, utils

PROCESS_EVENTS = True
INCLUDE_WINDOW_DATA = False

class CursorReplayStrategy(VanillaReplayStrategy):
    """Cursor replay strategy that allows the model to paint a red dot and self-correct.

    Extends the VanillaReplayStrategy to incorporate a visual cue (red dot) on the
    screenshot and provide an opportunity for self-correction based on the modified
    screenshot.

    If AGI or GPT6 happens, this script should be able to suddenly do the work.
        --LunjunZhang
    """
    def __init__(
        self,
        recording: models.Recording,
        replay_instructions: str = "",
        process_events: bool = PROCESS_EVENTS,,
        dot_radius: int = 5,
        dot_color: str = "red",
    ) -> None:
        """Initialize the CursorReplayStrategy.

        Args:
            recording (models.Recording): The recording object.
            replay_instructions (str): Natural language instructions
                for how recording should be replayed.
            process_events (bool): Flag indicating whether to process the events.
              Defaults to True.
            dot_radius (int): Radius of the dot to be painted.
            dot_color (str): Color of the dot to be painted.
        """
        super().__init__(recording, replay_instructions, process_events)
        self.dot_radius = dot_radius
        self.dot_color = dot_color
        self.action_history = []
        self.action_event_idx = 0

    def get_next_action_event(
        self,
        screenshot: models.Screenshot,
        window_event: models.WindowEvent,
    ) -> models.ActionEvent | None:
        """Get the next ActionEvent for replay with self-correction.

        Args:
            screenshot (models.Screenshot): The screenshot object.
            window_event (models.WindowEvent): The window event object.

        Returns:
            models.ActionEvent or None: The next ActionEvent for replay or None
              if there are no more events.
        """
        if self.process_events:
            action_events = self.recording.processed_action_events
        else:
            action_events = self.recording.action_events

        self.action_event_idx += 1
        num_action_events = len(action_events)
        if self.action_event_idx >= num_action_events:
            raise StopIteration()
        logger.debug(f"{self.action_event_idx=} of {num_action_events=}")

        action_event = generate_action_event_with_dot(
            screenshot,
            window_event,
            self.recording.action_events if not self.process_events else self.recording.processed_action_events,
            self.action_history,
            self.replay_instructions,
            self.dot_radius,
            self.dot_color,
        )
        
        if not action_event:
            raise StopIteration()

        self.action_history.append(action_event)
        return action_event

    def __del__(self) -> None:
        """Log the action history."""
        action_history_dicts = [
            action.to_prompt_dict() for action in self.action_history
        ]
        logger.info(f"action_history=\n{pformat(action_history_dicts)}")

def generate_action_event_with_dot(
    current_screenshot: models.Screenshot,
    current_window_event: models.WindowEvent,
    recorded_actions: list[models.ActionEvent],
    replayed_actions: list[models.ActionEvent],
    replay_instructions: str,
    dot_radius: int,
    dot_color: str,
) -> models.ActionEvent:
    """Modify the action event with the ability to paint a red dot and self-correct.

    Args:
        current_screenshot (models.Screenshot): current state screenshot
        current_window_event (models.WindowEvent): current state window data
        recorded_actions (list[models.ActionEvent]): list of action events from the recording
        replayed_actions (list[models.ActionEvent]): list of actions produced during current replay
        replay_instructions (str): proposed modifications in natural language instructions
        dot_radius (int): radius of the dot to be painted
        dot_color (str): color of the dot to be painted

    Returns:
        (models.ActionEvent) the next action event to be played, produced by the model
    """
    current_image = current_screenshot.image
    current_window_dict = current_window_event.to_prompt_dict()
    recorded_action_dicts = [action.to_prompt_dict() for action in recorded_actions]
    replayed_action_dicts = [action.to_prompt_dict() for action in replayed_actions]

    system_prompt = utils.render_template_from_file(
        "prompts/system_prompt_with_dot.j2",
    )
    prompt = utils.render_template_from_file(
        "prompts/action_event_generation_with_dot.j2",
        current_window=current_window_dict,
        recorded_actions=recorded_action_dicts,
        replayed_actions=replayed_action_dicts,
        replay_instructions=replay_instructions,
    )
    prompt_adapter = adapters.get_default_prompt_adapter()  
    content = prompt_adapter.prompt(
        prompt,
        system_prompt,
        [current_image],
    )
    action_dict = utils.parse_code_snippet(content)
    logger.info(f"{action_dict=}")
    if not action_dict:
        # allow early stopping
        return None
    
    image = Image.open(io.BytesIO(current_screenshot.image))
    draw = ImageDraw.Draw(image)
    x, y = action_dict['x'], action_dict['y']
    draw.ellipse(
        [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)],
        fill = dot_color,
        outline = dot_color,
    )
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_with_dot = buffer.getvalue()
    
    prompt_with_dot = utils.render_template_from_file(
        "prompts/self_correction_with_dot.j2",
        current_window=current_window_dict,
        recorded_actions=recorded_action_dicts,
        replayed_actions=replayed_action_dicts,
        replay_instructions=replay_instructions,
        initial_action=action_dict,
    )
    
    content_corrected = prompt_adapter.prompt(
        prompt_with_dot,
        system_prompt,
        [img_with_dot],
    )
    action_dict_corrected = utils.parse_code_snippet(content_corrected)
    
    if not action_dict_corrected:
        # Fallback to the original action
        action_dict_corrected = action_dict 
    
    action = models.ActionEvent.from_dict(action_dict_corrected)
    logger.info(f"{action=}")
    return action
   
