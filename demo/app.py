from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def main() -> None:
    st.set_page_config(page_title="Adaptive P2P Risk Demo", layout="wide")
    st.title("Adaptive P2P Risk")

    version = _api_get("/model/version")
    if version["ok"]:
        st.caption(
            f"Model source: {version['data'].get('selected_model_source')} | "
            f"DVC hash: {version['data'].get('dvc_output_hash') or 'unavailable'}"
        )
    else:
        st.error(version["error"])

    contract_tab, pool_tab, version_tab = st.tabs(["Contract", "Pool", "Model"])
    with contract_tab:
        _contract_form()
    with pool_tab:
        _pool_form()
    with version_tab:
        if version["ok"]:
            st.json(version["data"])
        else:
            st.warning("Model metadata is unavailable.")


def _contract_form() -> None:
    with st.form("contract-score-form"):
        left, right = st.columns(2)
        with left:
            exposure = st.number_input("Exposure", min_value=0.0, max_value=1.0, value=0.72)
            veh_power = st.number_input("VehPower", min_value=0.0, value=6.0)
            veh_age = st.number_input("VehAge", min_value=0.0, value=4.0)
            driv_age = st.number_input("DrivAge", min_value=0.0, value=42.0)
            bonus_malus = st.number_input("BonusMalus", min_value=0.0, value=68.0)
            density = st.number_input("Density", min_value=0.0, value=850.0)
        with right:
            pool_id = st.text_input("pool_id", value="0")
            veh_brand = st.text_input("VehBrand", value="B1")
            veh_gas = st.selectbox("VehGas", ["Regular", "Diesel"])
            area = st.text_input("Area", value="C")
            region = st.text_input("Region", value="R1")
            idpol = st.text_input("IDpol", value="")

        submitted = st.form_submit_button("Score Contract")
    if not submitted:
        return

    payload: dict[str, Any] = {
        "pool_id": pool_id,
        "Exposure": exposure,
        "VehPower": veh_power,
        "VehAge": veh_age,
        "DrivAge": driv_age,
        "BonusMalus": bonus_malus,
        "Density": density,
        "VehBrand": veh_brand,
        "VehGas": veh_gas,
        "Area": area,
        "Region": region,
    }
    if idpol:
        payload["IDpol"] = idpol
    response = _api_post("/score/contract", payload)
    if response["ok"]:
        probability = response["data"]["predicted_claim_probability"]
        st.metric("Claim Probability", f"{probability:.4f}")
        st.json(response["data"])
    else:
        st.error(response["error"])


def _pool_form() -> None:
    pool_id = st.text_input("Pool ID", value="0")
    default_members = pd.DataFrame(
        [
            {
                "pool_id": pool_id,
                "Exposure": 0.70,
                "VehPower": 6.0,
                "VehAge": 4.0,
                "DrivAge": 42.0,
                "BonusMalus": 68.0,
                "Density": 850.0,
                "VehBrand": "B1",
                "VehGas": "Regular",
                "Area": "C",
                "Region": "R1",
            },
            {
                "pool_id": pool_id,
                "Exposure": 0.35,
                "VehPower": 10.0,
                "VehAge": 11.0,
                "DrivAge": 27.0,
                "BonusMalus": 112.0,
                "Density": 2100.0,
                "VehBrand": "B2",
                "VehGas": "Diesel",
                "Area": "E",
                "Region": "R2",
            },
        ]
    )
    edited_members = st.data_editor(default_members, num_rows="dynamic", width="stretch")
    if not st.button("Score Pool"):
        return

    members = edited_members.fillna("").to_dict(orient="records")
    for member in members:
        member["pool_id"] = pool_id
    response = _api_post("/score/pool", {"pool_id": pool_id, "members": members})
    if response["ok"]:
        st.metric("Pool Risk Score", f"{response['data']['pool_risk_score']:.4f}")
        st.dataframe(pd.DataFrame(response["data"]["member_scores"]), width="stretch")
        st.json(response["data"])
    else:
        st.error(response["error"])


def _api_get(path: str) -> dict[str, Any]:
    return _api_request("GET", path)


def _api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _api_request("POST", path, payload)


def _api_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"ok": True, "data": json.loads(response.read().decode("utf-8"))}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8")
        return {"ok": False, "error": f"API returned {error.code}: {detail}"}
    except urllib.error.URLError as error:
        return {"ok": False, "error": f"API is unreachable at {API_BASE_URL}: {error.reason}"}


if __name__ == "__main__":
    main()
