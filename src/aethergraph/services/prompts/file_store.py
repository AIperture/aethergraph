# services/prompts/file_store.py
# Simple file-based prompt store
from pathlib import Path

class FilePromptStore:
    def __init__(self, root: str = "./prompts"):
        self.root = Path(root)

    async def get(self, name: str, version: str|None=None) -> str:
        """Get prompt by name and optional version.
        If version is None, get the latest (unversioned) prompt.
        
        Args:
            name: Prompt name (filename without extension)
            version: Optional version string
            Returns:
            Prompt content as string
        
        Example:
            prompt = await store.get("welcome_message", version="v1")
            print(prompt)
            # ./prompts/welcome_message@v1.md
        """
        p = self.root / (f"{name}@{version}.md" if version else f"{name}.md")
        return p.read_text(encoding="utf-8")
    
    async def render(self, name: str, **vars) -> str:
        """Get and render prompt with variable substitution.
        
        Args:
            name: Prompt name (filename without extension)
            **vars: Variables to substitute in the prompt
        Returns:
            Rendered prompt content as string
        """
        # Tiny {{var}} replacement; swap later for jinja/mustache
        txt = await self.get(name)
        for k,v in vars.items(): txt = txt.replace(f"{{{{{k}}}}}", str(v))
        return txt
