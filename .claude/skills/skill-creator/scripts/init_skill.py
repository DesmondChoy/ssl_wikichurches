#!/usr/bin/env python3
"""Initialize a new skill directory with template files."""

import argparse
import os
from pathlib import Path


def create_skill(skill_name: str, output_path: Path) -> None:
    """Create a new skill directory with template files."""
    skill_dir = output_path / skill_name

    if skill_dir.exists():
        print(f"Error: Directory already exists: {skill_dir}")
        return

    # Create directory structure
    skill_dir.mkdir(parents=True)
    (skill_dir / "scripts").mkdir()
    (skill_dir / "references").mkdir()
    (skill_dir / "assets").mkdir()

    # Create SKILL.md template
    skill_md = f'''---
name: {skill_name}
description: TODO - Describe what this skill does and when Claude should use it. Include specific triggers and contexts.
---

# {skill_name.replace("-", " ").title()}

TODO: Add instructions for using this skill.

## Overview

Describe the skill's purpose and capabilities.

## Usage

### Basic Usage

```bash
# Example commands or workflows
```

## Resources

- `scripts/` - Executable scripts (if any)
- `references/` - Reference documentation (if any)
- `assets/` - Templates and assets (if any)
'''
    (skill_dir / "SKILL.md").write_text(skill_md)

    # Create example script
    example_script = '''#!/usr/bin/env python3
"""Example script - customize or delete as needed."""


def main():
    print("Hello from the skill!")


if __name__ == "__main__":
    main()
'''
    (skill_dir / "scripts" / "example.py").write_text(example_script)

    # Create example reference
    example_ref = '''# Example Reference

This is an example reference file. Add domain-specific documentation here.

Delete this file if not needed.
'''
    (skill_dir / "references" / "example.md").write_text(example_ref)

    # Create .gitkeep in assets
    (skill_dir / "assets" / ".gitkeep").write_text("")

    print(f"Created skill at: {skill_dir}")
    print(f"\nNext steps:")
    print(f"  1. Edit {skill_dir}/SKILL.md")
    print(f"  2. Add scripts, references, and assets as needed")
    print(f"  3. Delete example files you don't need")
    print(f"  4. Run package_skill.py when ready")


def main():
    parser = argparse.ArgumentParser(description="Initialize a new skill directory")
    parser.add_argument("skill_name", help="Name of the skill (e.g., 'pdf-editor')")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)"
    )

    args = parser.parse_args()

    # Validate skill name
    if not args.skill_name.replace("-", "").replace("_", "").isalnum():
        print("Error: Skill name should only contain letters, numbers, hyphens, and underscores")
        return

    create_skill(args.skill_name, args.path)


if __name__ == "__main__":
    main()
