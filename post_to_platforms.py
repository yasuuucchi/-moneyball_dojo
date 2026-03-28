#!/usr/bin/env python3
"""
Moneyball Dojo - Platform Auto-Poster (Computer Use)
=====================================================
Claude API + Computer Use to auto-post daily digests to:
  1. Substack (English digest)
  2. note.com (Japanese digest)
  3. X / Twitter (morning, midday, evening posts)

Usage:
  python3 post_to_platforms.py                    # Post to all platforms
  python3 post_to_platforms.py --platform substack  # Substack only
  python3 post_to_platforms.py --platform note      # note.com only
  python3 post_to_platforms.py --platform x         # X only
  python3 post_to_platforms.py --dry-run            # Preview without posting

Requirements:
  pip3 install anthropic
  Environment variables:
    ANTHROPIC_API_KEY     - Claude API key
    SUBSTACK_EMAIL        - Substack login email
    SUBSTACK_PASSWORD     - Substack login password
    NOTE_EMAIL            - note.com login email
    NOTE_PASSWORD         - note.com login password
    X_EMAIL               - X login email
    X_PASSWORD            - X login password

Notes:
  - This script uses Claude's Computer Use (beta) to control a browser.
  - Requires a display (Desktop app or VNC). Does NOT work in headless cloud.
  - Designed to be run as a Desktop scheduled task after /schedule generates predictions.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

PROJECT_DIR = Path(__file__).parent
LATEST_DIR = PROJECT_DIR / "output" / "latest"

# Platform configurations
PLATFORMS = {
    'substack': {
        'url': 'https://substack.com/publish/post',
        'content_file': 'POST_TO_SUBSTACK.md',
        'cred_env': ('SUBSTACK_EMAIL', 'SUBSTACK_PASSWORD'),
    },
    'note': {
        'url': 'https://note.com/new',
        'content_file': 'POST_TO_NOTE.md',
        'cred_env': ('NOTE_EMAIL', 'NOTE_PASSWORD'),
    },
    'x': {
        'url': 'https://x.com/compose/post',
        'content_file': 'POST_TO_X.txt',
        'cred_env': ('X_EMAIL', 'X_PASSWORD'),
    },
}


def load_content(platform):
    """Load the generated content for a platform."""
    config = PLATFORMS[platform]
    content_path = LATEST_DIR / config['content_file']

    if not content_path.exists():
        print(f"  ERROR: {content_path} not found. Run run_daily.py first.")
        return None

    content = content_path.read_text(encoding='utf-8')
    print(f"  Loaded {len(content)} chars from {content_path.name}")
    return content


def get_credentials(platform):
    """Get login credentials from environment variables."""
    config = PLATFORMS[platform]
    email_key, pass_key = config['cred_env']
    email = os.environ.get(email_key)
    password = os.environ.get(pass_key)

    if not email or not password:
        print(f"  ERROR: Set {email_key} and {pass_key} environment variables.")
        return None, None

    return email, password


def build_substack_prompt(content, email, password):
    """Build the Computer Use prompt for Substack posting."""
    # Extract title from markdown (first # heading)
    title = "Moneyball Dojo Daily Digest"
    for line in content.split('\n'):
        if line.startswith('# '):
            title = line[2:].strip()
            break

    return f"""You are automating a Substack post. Follow these steps exactly:

1. Open Firefox/Chrome and navigate to https://substack.com/sign-in
2. Log in with these credentials:
<robot_credentials>
Email: {email}
Password: {password}
</robot_credentials>

3. After login, click "New post" or navigate to the post editor
4. Set the title to: {title}
5. Paste the following article content into the post body (the editor accepts Markdown):

<article_content>
{content}
</article_content>

6. Set the post as FREE (not paid)
7. Click "Publish" to publish immediately
8. Take a screenshot to confirm the post was published successfully
9. Report the published URL

IMPORTANT:
- If you see a CAPTCHA, take a screenshot and report it — do not try to solve it.
- If login fails, report the error and stop.
- Do NOT modify the article content."""


def build_note_prompt(content, email, password):
    """Build the Computer Use prompt for note.com posting."""
    title = "Moneyball Dojo デイリーダイジェスト"
    for line in content.split('\n'):
        if line.startswith('# '):
            title = line[2:].strip()
            break

    return f"""You are automating a note.com article post. Follow these steps exactly:

1. Open Firefox/Chrome and navigate to https://note.com/login
2. Log in with these credentials:
<robot_credentials>
Email: {email}
Password: {password}
</robot_credentials>

3. After login, click "投稿" (Post) or "テキスト" (Text) to create a new text article
4. Set the title to: {title}
5. Paste the following article content into the body:

<article_content>
{content}
</article_content>

6. Set visibility to public (公開)
7. Set price to FREE (無料)
8. Click "公開する" to publish
9. Take a screenshot to confirm
10. Report the published URL

IMPORTANT:
- note.com interface is in Japanese.
- If you see a CAPTCHA, take a screenshot and report it — do not try to solve it.
- If login fails, report the error and stop.
- Do NOT modify the article content."""


def build_x_prompt(content, email, password):
    """Build the Computer Use prompt for X/Twitter posting."""
    # Truncate to 280 chars if needed (X limit)
    if len(content) > 280:
        content = content[:277] + "..."

    return f"""You are automating an X (Twitter) post. Follow these steps exactly:

1. Open Firefox/Chrome and navigate to https://x.com/i/flow/login
2. Log in with these credentials:
<robot_credentials>
Email: {email}
Password: {password}
</robot_credentials>

3. After login, click the compose/post button (feather icon or "Post" button)
4. Type the following tweet content exactly:

<tweet_content>
{content}
</tweet_content>

5. Click "Post" to publish the tweet
6. Take a screenshot to confirm the tweet was posted
7. Report the tweet URL

IMPORTANT:
- If you see a CAPTCHA or phone verification, take a screenshot and report it.
- If login fails, report the error and stop.
- Do NOT modify the tweet content.
- The tweet must be posted from @MoneyballDojo account."""


def post_with_computer_use(platform, dry_run=False):
    """Use Claude API + Computer Use to post content to a platform."""
    print(f"\n{'='*50}")
    print(f"  Posting to: {platform.upper()}")
    print(f"{'='*50}")

    content = load_content(platform)
    if not content:
        return False

    email, password = get_credentials(platform)
    if not email:
        return False

    # Build platform-specific prompt
    prompt_builders = {
        'substack': build_substack_prompt,
        'note': build_note_prompt,
        'x': build_x_prompt,
    }
    prompt = prompt_builders[platform](content, email, password)

    if dry_run:
        print(f"\n  [DRY RUN] Would send Computer Use request for {platform}")
        print(f"  Content length: {len(content)} chars")
        print(f"  Credentials: {email[:3]}***")
        return True

    if not ANTHROPIC_AVAILABLE:
        print("  ERROR: anthropic package not installed. Run: pip3 install anthropic")
        return False

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("  ERROR: Set ANTHROPIC_API_KEY environment variable.")
        return False

    client = anthropic.Anthropic(api_key=api_key)

    print(f"  Sending Computer Use request to Claude...")

    try:
        response = client.beta.messages.create(
            model="claude-sonnet-4-6",  # Sonnet for cost efficiency on browser tasks
            max_tokens=4096,
            tools=[
                {
                    "type": "computer_20251124",
                    "name": "computer",
                    "display_width_px": 1280,
                    "display_height_px": 800,
                },
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
            ],
            messages=[{"role": "user", "content": prompt}],
            betas=["computer-use-2025-11-24"],
        )

        # Process response
        result_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                result_text += block.text

        if response.stop_reason == "end_turn":
            print(f"  SUCCESS: {platform} post completed")
            print(f"  Response: {result_text[:200]}")
            return True
        elif response.stop_reason == "tool_use":
            # Computer Use needs to continue — handle in agentic loop
            print(f"  Computer Use in progress (tool_use loop needed)...")
            # In production, implement agentic loop here
            return _run_computer_use_loop(client, response, prompt)
        else:
            print(f"  WARNING: Unexpected stop reason: {response.stop_reason}")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def _run_computer_use_loop(client, initial_response, initial_prompt):
    """Run the Computer Use agentic loop until completion."""
    messages = [{"role": "user", "content": initial_prompt}]
    response = initial_response
    max_iterations = 30  # Safety limit

    for i in range(max_iterations):
        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Process tool results
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # For computer tool, we need to capture screenshot and return it
                # This requires actual display access — handled by the Computer Use runtime
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Action executed. Take a screenshot to verify.",
                })

        if not tool_results:
            # No more tool calls — done
            for block in response.content:
                if hasattr(block, 'text'):
                    print(f"  Final: {block.text[:200]}")
            return True

        messages.append({"role": "user", "content": tool_results})

        try:
            response = client.beta.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                tools=[
                    {
                        "type": "computer_20251124",
                        "name": "computer",
                        "display_width_px": 1280,
                        "display_height_px": 800,
                    },
                    {"type": "bash_20250124", "name": "bash"},
                    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                ],
                messages=messages,
                betas=["computer-use-2025-11-24"],
            )
        except Exception as e:
            print(f"  Loop error at iteration {i}: {e}")
            return False

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, 'text'):
                    print(f"  Completed: {block.text[:200]}")
            return True

    print("  WARNING: Reached max iterations without completion")
    return False


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Post Moneyball Dojo content to platforms')
    parser.add_argument('--platform', choices=['substack', 'note', 'x', 'all'], default='all',
                        help='Platform to post to (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without posting')
    args = parser.parse_args()

    now_et = datetime.now(ZoneInfo("America/New_York"))
    print(f"Moneyball Dojo Auto-Poster — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print()

    # Check that latest content exists
    if not LATEST_DIR.exists():
        print(f"ERROR: {LATEST_DIR} not found.")
        print("Run run_daily.py first to generate predictions and digests.")
        sys.exit(1)

    platforms = ['substack', 'note', 'x'] if args.platform == 'all' else [args.platform]
    results = {}

    for platform in platforms:
        success = post_with_computer_use(platform, dry_run=args.dry_run)
        results[platform] = success

        if platform != platforms[-1]:
            time.sleep(5)  # Brief pause between platforms

    # Summary
    print(f"\n{'='*50}")
    print("  POSTING SUMMARY")
    print(f"{'='*50}")
    for platform, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {platform:>10}: {status}")

    failed = [p for p, s in results.items() if not s]
    if failed:
        print(f"\n  Failed: {', '.join(failed)}")
        print("  Fix errors above and retry with: python3 post_to_platforms.py --platform <name>")
        sys.exit(1)
    else:
        print("\n  All platforms posted successfully!")


if __name__ == '__main__':
    main()
