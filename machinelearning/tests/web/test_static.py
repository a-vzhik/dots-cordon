from fastapi.testclient import TestClient

from dots_cordon_ml.audit.web.app import create_app


def test_dashboard_and_assets_are_served_without_opening_database(tmp_path):
    static = tmp_path / "static"
    assets = static / "assets"
    assets.mkdir(parents=True)
    (static / "index.html").write_text(
        '<!doctype html><title>Training workspace</title><script src="/assets/app.js"></script>'
    )
    (assets / "app.js").write_text("window.auditDashboard = true;")
    database = tmp_path / "missing.sqlite3"
    with TestClient(create_app(f"sqlite:///{database}", frontend_dir=static)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-cache"
        assert "/assets/app.js" in response.text
        assert client.get("/assets/app.js").text == "window.auditDashboard = true;"
        assert client.get("/assets/..%2Findex.html").status_code == 404
        assert client.get("/api/v1/not-a-route").status_code == 404
        assert client.post("/", json={}).status_code == 405
        assert len(client.get("/openapi.json").json()["paths"]) == 9
    assert not database.exists()


def test_missing_frontend_returns_build_instructions_but_api_still_works(tmp_path):
    with TestClient(
        create_app(
            f"sqlite:///{tmp_path / 'missing.sqlite3'}",
            frontend_dir=tmp_path / "not-built",
        )
    ) as client:
        response = client.get("/")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "frontend_not_built"
        assert "npm run build" in response.json()["error"]["message"]
        assert client.get("/docs").status_code == 200
        assert client.get("/health").status_code == 503
