#!/usr/bin/env python3
"""Package a skill into a distributable .skill file."""

import argparse
import re
import shutil
import sys
from pathlib import Path


def validate_skill(skill_dir: Path) -> list[str]:
    """Validate a skill directory. Returns list of errors."""
    errors = []

    # Check SKILL.md exists
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        errors.append("Missing required SKILL.md file")
        return errors  # Can't continue without SKILL.md

    content = skill_md.read_text()

    # Check frontmatter
    if not content.startswith("---"):
        errors.append("SKILL.md must start with YAML frontmatter (---)")
        return errors

    # Extract frontmatter
    frontmatter_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        errors.append("Invalid YAML frontmatter format")
        return errors

    frontmatter = frontmatter_match.group(1)

    # Check required fields
    if "name:" not in frontmatter:
        errors.append("Frontmatter missing required 'name' field")
    if "description:" not in frontmatter:
        errors.append("Frontmatter missing required 'description' field")

    # Check description quality
    desc_match = re.search(r"description:\s*(.+?)(?:\n[a-z]+:|$)", frontmatter, re.DOTALL)
    if desc_match:
        description = desc_match.group(1).strip()
        if "TODO" in description:
            errors.append("Description contains TODO placeholder")
        if len(description) < 50:
            errors.append("Description is too short (should be at least 50 characters)")

    # Check body content
    body_start = content.find("---", 3) + 3
    body = content[body_start:].strip()

    if len(body) < 100:
        errors.append("SKILL.md body is too short (should be at least 100 characters)")

    if "TODO" in body:
        errors.append("SKILL.md body contains TODO placeholders")

    # Check for forbidden files
    forbidden = ["README.md", "CHANGELOG.md", "INSTALLATION_GUIDE.md", "QUICK_REFERENCE.md"]
    for fname in forbidden:
        if (skill_dir / fname).exists():
            errors.append(f"Skill contains forbidden file: {fname}")

    return errors


def package_skill(skill_dir: Path, output_dir: Path) -> Path | None:
    """Package a skill directory into a .skill file."""
    # Validate first
    errors = validate_skill(skill_dir)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return None

    # Create package
    skill_name = skill_dir.name
    output_file = output_dir / f"{skill_name}.skill"

    # Create zip (without .skill extension first, then rename)
    temp_zip = output_dir / skill_name
    shutil.make_archive(str(temp_zip), "zip", skill_dir.parent, skill_dir.name)

    # Rename to .skill
    zip_file = output_dir / f"{skill_name}.zip"
    if output_file.exists():
        output_file.unlink()
    zip_file.rename(output_file)

    print(f"Successfully created: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Package a skill into a .skill file")
    parser.add_argument("skill_path", type=Path, help="Path to the skill directory")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Output directory (default: current directory)"
    )

    args = parser.parse_args()

    if not args.skill_path.is_dir():
        print(f"Error: Not a directory: {args.skill_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = package_skill(args.skill_path, args.output_dir)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
