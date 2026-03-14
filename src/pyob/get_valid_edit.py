import difflib
import os
import re
import time

from .core_utils import logger


class GetValidEditMixin:
    def get_valid_edit(
        self,
        prompt: str,
        source_code: str,
        require_edit: bool = True,
        target_filepath: str = "",
    ) -> tuple[str, str, str]:
        display_name = (
            os.path.relpath(target_filepath, getattr(self, "target_dir"))
            if target_filepath
            else "System Update"
        )

        # 1. Pre-Flight Human Check
        prompt, skip = self._handle_pre_generation_approval(prompt, display_name)
        if skip:
            return source_code, "AI generation skipped by user.", ""

        attempts = 0
        while True:
            # 2. Fetch from AI (Handles keys, retries, and API limits)
            response_text, attempts = self._fetch_llm_with_retries(
                prompt, display_name, attempts
            )

            # 3. Validate and Apply XML Patch
            new_code, explanation, is_valid = self._validate_llm_patch(
                source_code, response_text, require_edit, display_name
            )

            if not is_valid:
                attempts += 1
                continue

            if new_code == source_code:
                return new_code, explanation, response_text

            # 4. Post-Flight Human Review (Diffs and Approval)
            final_code, final_exp, final_resp, action = (
                self._handle_post_generation_review(
                    source_code,
                    new_code,
                    explanation,
                    response_text,
                    target_filepath,
                    display_name,
                )
            )

            if action == "REGENERATE":
                attempts += 1
                continue

            return final_code, final_exp, final_resp

    # ==========================================
    # PRIVATE HELPER METHODS
    # ==========================================

    def _handle_pre_generation_approval(
        self, prompt: str, display_name: str
    ) -> tuple[str, bool]:
        print("\n" + "=" * 50)
        print(f"AI Generation Prompt Ready: [{display_name}]")
        print("=" * 50)
        choice = getattr(self, "get_user_approval")(
            "Hit ENTER to send as-is, type 'EDIT_PROMPT', 'AUGMENT_PROMPT', or 'SKIP'.",
            timeout=220,
        )
        if choice == "SKIP":
            return prompt, True
        elif choice == "EDIT_PROMPT":
            prompt = getattr(self, "_edit_prompt_with_external_editor")(prompt)
        elif choice == "AUGMENT_PROMPT":
            aug = getattr(self, "_get_user_prompt_augmentation")()
            if aug.strip():
                prompt += f"\n\n### User Augmentation:\n{aug.strip()}"
        return prompt, False

    def _fetch_llm_with_retries(
        self, prompt: str, display_name: str, attempts: int
    ) -> tuple[str, int]:
        is_cloud = (
            os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("CI") == "true"
            or "GITHUB_RUN_ID" in os.environ
        )
        key_cooldowns = getattr(self, "key_cooldowns", {})

        while True:
            now = time.time()
            gemini_keys = [k for k in key_cooldowns.keys() if "github" not in k]
            available_keys = [k for k in gemini_keys if now > key_cooldowns[k]]
            key = None
            gh_model = "Llama-3"

            if available_keys:
                key = available_keys[attempts % len(available_keys)]
                logger.info(
                    f"\n[Attempting Gemini API Key {attempts % len(available_keys) + 1}/{len(gemini_keys)}]"
                )
                response = getattr(self, "_stream_single_llm")(
                    prompt, key=key, context=display_name
                )
            elif is_cloud:
                if now < key_cooldowns.get("github_llama", 0):
                    gh_model = "Phi-4"
                if gh_model == "Phi-4" and now < key_cooldowns.get("github_phi", 0):
                    wait = 60
                    logger.warning(
                        f"All API limits exhausted. Sleeping {wait}s for Gemini refill..."
                    )
                    time.sleep(wait)
                    attempts += 1
                    continue

                logger.warning(
                    f"Gemini limited. Pivoting to GitHub Models ({gh_model})..."
                )
                response = getattr(self, "_stream_single_llm")(
                    prompt, key=None, context=display_name, gh_model=gh_model
                )
            else:
                logger.info("\n[All keys exhausted. Falling back to Local Ollama]")
                response = getattr(self, "_stream_single_llm")(
                    prompt, key=None, context=display_name
                )

            if response.startswith("ERROR_CODE_429"):
                if key:
                    key_cooldowns[key] = time.time() + 120
                    logger.warning("Key rate limited. Pivoting to next key...")
                    attempts += 1
                    continue
                elif "RateLimitReached" in response:
                    import re

                    match = re.search(r"wait (\d+) seconds", response)
                    wait_time = int(match.group(1)) if match else 86400
                    if gh_model == "Llama-3":
                        key_cooldowns["github_llama"] = time.time() + wait_time + 60
                    else:
                        key_cooldowns["github_phi"] = time.time() + wait_time + 60
                    logger.error(
                        f"GITHUB QUOTA REACHED ({gh_model}). Cooldown: {wait_time}s"
                    )
                    attempts += 1
                    continue
                else:
                    time.sleep(60)
                    attempts += 1
                    continue

            if "ERROR_CODE_413" in response:
                logger.warning("Payload too large (413). Backing off 60s...")
                time.sleep(60)
                attempts += 1
                continue

            if response.startswith("ERROR_CODE_") or not response.strip():
                if key and "429" not in (response or ""):
                    key_cooldowns[key] = time.time() + 10

                if available_keys:
                    logger.warning(f"Engine failed with error: {str(response)[:60]}... Rotating...")
                    attempts += 1
                    time.sleep(2)
                    continue

                logger.warning("API Error or Empty Response. Sleeping 60s...")
                time.sleep(60)
                attempts += 1
                continue

            return response, attempts

    def _validate_llm_patch(
        self,
        source_code: str,
        response_text: str,
        require_edit: bool,
        display_name: str,
    ) -> tuple[str, str, bool]:
        new_code, explanation, edit_success = getattr(self, "apply_xml_edits")(
            source_code, response_text
        )
        edit_count = len(re.findall(r"<EDIT>", response_text, re.IGNORECASE))
        lower_exp = explanation.lower()
        ai_approved = "no fixes needed" in lower_exp or "looks good" in lower_exp

        if not require_edit and ai_approved:
            return source_code, explanation, True
        if edit_count > 0 and not edit_success:
            logger.warning(
                f"Partial edit failure in {display_name}. Auto-regenerating..."
            )
            time.sleep(30)
            return source_code, explanation, False
        if require_edit and new_code == source_code:
            logger.warning("Search block mismatch. Rotating...")
            time.sleep(30)
            return source_code, explanation, False
        if not require_edit and new_code == source_code and not ai_approved:
            time.sleep(30)
            return source_code, explanation, False

        return new_code, explanation, True

    def _handle_post_generation_review(
        self,
        source_code: str,
        new_code: str,
        explanation: str,
        response_text: str,
        target_filepath: str,
        display_name: str,
    ) -> tuple[str, str, str, str]:
        print("\n" + "=" * 50)
        print(f"AI Proposed Edit Ready for: [{display_name}]")
        print("=" * 50)
        diff_lines = list(
            difflib.unified_diff(
                source_code.splitlines(keepends=True),
                new_code.splitlines(keepends=True),
                fromfile="Original",
                tofile="Proposed",
            )
        )
        for line in diff_lines[2:22]:
            clean = line.rstrip()
            if clean.startswith("+"):
                print(f"\033[92m{clean}\033[0m")
            elif clean.startswith("-"):
                print(f"\033[91m{clean}\033[0m")
            elif clean.startswith("@@"):
                print(f"\033[94m{clean}\033[0m")
            else:
                print(clean)

        choice = getattr(self, "get_user_approval")(
            "Hit ENTER to APPLY, type 'EDIT_CODE', 'EDIT_XML', 'REGENERATE', or 'SKIP'.",
            timeout=220,
        )

        if choice == "SKIP":
            return source_code, "Edit skipped.", "", "SKIP"
        elif choice == "REGENERATE":
            return source_code, explanation, response_text, "REGENERATE"
        elif choice == "EDIT_XML":
            resp = getattr(self, "_edit_prompt_with_external_editor")(response_text)
            nc, exp, _ = getattr(self, "apply_xml_edits")(source_code, resp)
            return nc, exp, resp, "APPLY"
        elif choice == "EDIT_CODE":
            ext = os.path.splitext(target_filepath)[1] if target_filepath else ".py"
            ec = getattr(self, "_launch_external_code_editor")(
                new_code, file_suffix=ext
            )
            return ec, explanation + " (User edited)", response_text, "APPLY"

        return new_code, explanation, response_text, "APPLY"
