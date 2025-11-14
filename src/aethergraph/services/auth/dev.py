# services/auth/dev.py
class DevTokenAuthn:
    """Development token authenticator. Accepts any token, returns 'dev' as subject."""

    def __init__(self, header="x-dev-token"):
        self.header = header

    async def whoami(self, token: str | None) -> dict:
        return {"subject": token or "dev", "roles": ["admin"]}


class AllowAllAuthz:
    """Development authorizer that allows all actions."""

    async def allow(self, actor: dict, action: str, resource: str) -> bool:
        return True
