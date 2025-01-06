import datetime
import json
import logging
import os

logger = logging.getLogger("desktopenv.experiment")


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir):
    logger.info("=== Starting new task ===")
    logger.info(f"Instruction: {instruction}")
    logger.info(f"Max steps allowed: {max_steps}")

    agent.reset()
    obs = env.reset(task_config=example)
    done = False
    step_idx = 0
    env.controller.start_recording()

    while not done and step_idx < max_steps:
        logger.info(f"\n=== Step {step_idx + 1} ===")

        # Get agent's response
        response, actions = agent.predict(instruction, obs)
        if response:
            logger.info(f"Agent reasoning: {response}")
        else:
            logger.warning("Agent did not provide reasoning")

        if not actions:
            logger.error("Agent failed to generate actions")
            break

        logger.info(f"Generated {len(actions)} actions to try")

        for action_idx, action in enumerate(actions, 1):
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

            logger.info(f"Executing action {
                        action_idx}/{len(actions)}: {action}")

            # Execute action and get results
            obs, reward, done, info = env.step(
                action, args.sleep_after_execution)

            # Save screenshot
            screenshot_filename = f"step_{step_idx + 1}_{action_timestamp}.png"
            with open(os.path.join(example_result_dir, screenshot_filename), "wb") as _f:
                _f.write(obs['screenshot'])

            # Log trajectory
            trajectory_entry = {
                "step_num": step_idx + 1,
                "action_timestamp": action_timestamp,
                "action": action,
                "done": done,
                "screenshot_file": screenshot_filename
            }

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps(trajectory_entry))
                f.write("\n")

            if done:
                logger.info("✓ Task completed successfully!")
                break

        step_idx += 1
        if step_idx >= max_steps:
            logger.warning("Reached maximum steps without completing task")

    # Save recording
    env.controller.end_recording(os.path.join(
        example_result_dir, "recording.mp4"))
    logger.info(f"\n=== Task Complete ===")
    logger.info(f"Steps taken: {step_idx}/{max_steps}")
