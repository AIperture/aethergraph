class DevTokenAuthn:
    """Development token authenticator. Accepts any token, returns 'dev' as subject."""

    def __init__(self, header="x-dev-token"):
        self.header = header

    async def whoami(self, token: str | None) -> dict:
        return {"subject": token or "dev", "roles": ["admin"]}
