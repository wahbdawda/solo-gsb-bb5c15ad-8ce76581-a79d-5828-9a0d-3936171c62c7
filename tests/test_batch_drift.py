"""GET /batches/{batch_id}/drift：批次漂移审计（只读）端到端测试。"""

from __future__ import annotations


def _m(mid, irt=None, refs=None, sent="2026-09-18T10:00:00+00:00", subject="hello",
       retain_until=None):
    msg = {
        "message_id": mid,
        "in_reply_to": irt,
        "references": refs or [],
        "sent_at": sent,
        "subject": subject,
    }
    if retain_until is not None:
        msg["retain_until"] = retain_until
    return msg


def _drift_items(body):
    return {it["message_id"]: it for it in body["results"]}


def test_unknown_batch_returns_404(client):
    r = client.get("/batches/nope/drift")
    assert r.status_code == 404
    assert r.json()["detail"] == "batch not found"


def test_fresh_batch_all_nodes_unchanged_in_original_preorder(client):
    # 原始先序：P, Q；Z, A, C, D（根按 (sent_at, message_id)，A 树先序 A C D）
    batch = client.post("/batches", json={"messages": [
        _m("A", sent="2026-09-18T08:00:00+00:00"),
        _m("C", irt="A", refs=["A"], sent="2026-09-18T09:00:00+00:00"),
        _m("D", irt="C", refs=["A", "C"], sent="2026-09-18T10:00:00+00:00"),
        _m("P", sent="2026-09-18T07:00:00+00:00"),
        _m("Q", irt="P", refs=["P"], sent="2026-09-18T07:30:00+00:00"),
        _m("Z", irt="ghost", refs=["ghost"], sent="2026-09-18T12:00:00+00:00"),
    ]}).json()

    r = client.get(f"/batches/{batch['batch_id']}/drift")
    assert r.status_code == 200
    body = r.json()
    assert body["batch_id"] == batch["batch_id"]

    # 仅审计快照 threads 节点，按原始先序输出
    assert [it["message_id"] for it in body["results"]] == ["P", "Q", "A", "C", "D", "Z"]
    assert body["summary"] == {
        "total": 6, "deleted": 0, "id_reused": 0,
        "reparented": 0, "root_changed": 0, "unchanged": 6,
    }
    for it in body["results"]:
        assert it["status"] == "unchanged"
        assert it["current_parent_id"] == it["original_parent_id"]
        assert it["current_root_id"] == it["original_root_id"]

    p = _drift_items(body)["P"]
    assert p["original_parent_id"] is None
    assert p["current_parent_id"] is None
    assert p["original_root_id"] == p["current_root_id"] == "P"
    c = _drift_items(body)["C"]
    assert c["original_parent_id"] == "A"
    assert c["original_root_id"] == "A"


def test_thread_correction_detaching_subtree_marks_reparented_and_root_changed(
    client,
):
    # A -> C -> D；纠偏把 C（连同子树 D）拆成新根：
    # C 父 A -> None（reparented）；D 父仍为 C 但根 A -> C（root_changed）
    batch = client.post("/batches", json={"messages": [
        _m("A", sent="2026-09-18T08:00:00+00:00"),
        _m("C", irt="A", refs=["A"], sent="2026-09-18T09:00:00+00:00"),
        _m("D", irt="C", refs=["A", "C"], sent="2026-09-18T10:00:00+00:00"),
    ]}).json()
    rev = client.get("/threads/D").json()["revision"]

    r = client.post("/thread-corrections", json={
        "message_id": "C",
        "new_parent_id": None,
        "expected_revision": rev,
    })
    assert r.status_code == 200

    body = client.get(f"/batches/{batch['batch_id']}/drift").json()
    items = _drift_items(body)
    # A 仍是根，未参与移动 -> unchanged
    assert items["A"]["status"] == "unchanged"
    assert items["A"]["current_root_id"] == "A"
    # C 父指针 A -> None，根 A -> C -> reparented
    assert items["C"]["status"] == "reparented"
    assert items["C"]["original_parent_id"] == "A"
    assert items["C"]["current_parent_id"] is None
    assert items["C"]["original_root_id"] == "A"
    assert items["C"]["current_root_id"] == "C"
    # D 父指针仍是 C，但所在根 A -> C -> root_changed（不是 reparented）
    assert items["D"]["status"] == "root_changed"
    assert items["D"]["original_parent_id"] == items["D"]["current_parent_id"] == "C"
    assert items["D"]["original_root_id"] == "A"
    assert items["D"]["current_root_id"] == "C"
    assert body["summary"] == {
        "total": 3, "deleted": 0, "id_reused": 0,
        "reparented": 1, "root_changed": 1, "unchanged": 1,
    }

    # 再把 D 从 C 下移回 A 下：父 C -> A -> reparented（优先于 root_changed；
    # 此时 D 的根也由 C 变回其快照原始根 A）
    rev2 = client.get("/threads/D").json()["revision"]
    r2 = client.post("/thread-corrections", json={
        "message_id": "D",
        "new_parent_id": "A",
        "expected_revision": rev2,
    })
    assert r2.status_code == 200

    body2 = client.get(f"/batches/{batch['batch_id']}/drift").json()
    items2 = _drift_items(body2)
    assert items2["D"]["status"] == "reparented"
    assert items2["D"]["original_parent_id"] == "C"
    assert items2["D"]["current_parent_id"] == "A"
    assert items2["D"]["original_root_id"] == "A"
    assert items2["D"]["current_root_id"] == "A"
    # C 仍为 reparented；A 仍 unchanged
    assert items2["C"]["status"] == "reparented"
    assert items2["A"]["status"] == "unchanged"
    assert body2["summary"] == {
        "total": 3, "deleted": 0, "id_reused": 0,
        "reparented": 2, "root_changed": 0, "unchanged": 1,
    }


def test_relink_marks_broken_chain_node_reparented_and_root_changed(client):
    # D 归档时父 late 未到达 -> D 断链为根；补链后 D 挂到 late 下
    batch = client.post("/batches", json={"messages": [
        _m("D", irt="late", refs=["ghost"], sent="2026-09-18T09:00:00+00:00"),
    ]}).json()
    client.post("/batches", json={"messages": [
        _m("late", sent="2026-09-18T07:00:00+00:00"),
    ]})
    r = client.post("/relinks", json={"message_ids": ["D"]})
    assert r.status_code == 200
    assert r.json()["results"][0]["new_parent_id"] == "late"

    body = client.get(f"/batches/{batch['batch_id']}/drift").json()
    item = body["results"][0]
    assert item["message_id"] == "D"
    assert item["status"] == "reparented"
    assert item["original_parent_id"] is None
    assert item["current_parent_id"] == "late"
    assert item["original_root_id"] == "D"
    assert item["current_root_id"] == "late"
    assert body["summary"] == {
        "total": 1, "deleted": 0, "id_reused": 0,
        "reparented": 1, "root_changed": 0, "unchanged": 0,
    }


def test_cleanup_marks_deleted_reparented_and_root_changed(client):
    # R(到期) -> A(到期) -> C(未到期) -> E(retain 更晚)；R -> D(到期)
    batch = client.post("/batches", json={"messages": [
        _m("R", sent="2026-09-18T08:00:00+00:00",
           retain_until="2026-09-19T00:00:00+00:00"),
        _m("A", irt="R", refs=["R"], sent="2026-09-18T09:00:00+00:00",
           retain_until="2026-09-19T00:00:00+00:00"),
        _m("C", irt="A", refs=["R", "A"], sent="2026-09-18T10:00:00+00:00"),
        _m("E", irt="C", refs=["C"], sent="2026-09-18T10:30:00+00:00",
           retain_until="2026-10-01T00:00:00+00:00"),
        _m("D", irt="R", refs=["R"], sent="2026-09-18T11:00:00+00:00",
           retain_until="2026-09-19T00:00:00+00:00"),
    ]}).json()

    body = client.post("/retention/cleanups", json={
        "cutoff": "2026-09-20T00:00:00+00:00",
    }).json()
    assert body["deleted_ids"] == ["A", "D", "R"]

    audit = client.get(f"/batches/{batch['batch_id']}/drift").json()
    # 原始先序保持不变：R A C E D
    assert [it["message_id"] for it in audit["results"]] == ["R", "A", "C", "E", "D"]
    items = _drift_items(audit)

    # 被清理删除的节点：current 全部为 null
    for mid in ("R", "A", "D"):
        assert items[mid]["status"] == "deleted"
        assert items[mid]["current_parent_id"] is None
        assert items[mid]["current_root_id"] is None
    assert items["R"]["original_root_id"] == "R"
    assert items["D"]["original_parent_id"] == "R"

    # C 越过被删的 A 重挂为新根：父 A -> None（reparented），根 R -> C
    assert items["C"]["status"] == "reparented"
    assert items["C"]["original_parent_id"] == "A"
    assert items["C"]["current_parent_id"] is None
    assert items["C"]["original_root_id"] == "R"
    assert items["C"]["current_root_id"] == "C"

    # E 父仍为 C（未变），但根 R -> C -> root_changed
    assert items["E"]["status"] == "root_changed"
    assert items["E"]["original_parent_id"] == items["E"]["current_parent_id"] == "C"
    assert items["E"]["original_root_id"] == "R"
    assert items["E"]["current_root_id"] == "C"

    assert audit["summary"] == {
        "total": 5, "deleted": 3, "id_reused": 0,
        "reparented": 1, "root_changed": 1, "unchanged": 0,
    }


def test_same_id_rearchived_after_cleanup_is_id_reused_and_not_deleted(client):
    # R(到期) -> A(到期) -> C(未到期)
    batch1 = client.post("/batches", json={"messages": [
        _m("R", sent="2026-09-18T08:00:00+00:00",
           retain_until="2026-09-19T00:00:00+00:00"),
        _m("A", irt="R", refs=["R"], sent="2026-09-18T09:00:00+00:00",
           retain_until="2026-09-19T00:00:00+00:00"),
        _m("C", irt="A", refs=["R", "A"], sent="2026-09-18T10:00:00+00:00"),
    ]}).json()

    # 删除前：全部 unchanged
    before = client.get(f"/batches/{batch1['batch_id']}/drift").json()
    assert before["summary"]["unchanged"] == 3

    # 清理 R/A；C 提升为新根
    client.post("/retention/cleanups", json={"cutoff": "2026-09-20T00:00:00+00:00"})
    assert client.get("/messages/R").status_code == 404
    assert client.get("/messages/A").status_code == 404

    # 同 ID 重新归档（新批次）：R 重新自成根，A 挂到 R 下
    batch2 = client.post("/batches", json={"messages": [
        _m("R", sent="2026-10-18T08:00:00+00:00"),
        _m("A", irt="R", refs=["R"], sent="2026-10-18T09:00:00+00:00"),
    ]}).json()
    assert batch2["batch_id"] != batch1["batch_id"]

    # 以旧批次快照为基线：R/A 当前属于 batch2 -> id_reused（不再是 deleted）
    audit = client.get(f"/batches/{batch1['batch_id']}/drift").json()
    items = _drift_items(audit)
    assert items["R"]["status"] == "id_reused"
    assert items["R"]["current_parent_id"] is None
    assert items["R"]["current_root_id"] == "R"
    assert items["A"]["status"] == "id_reused"
    assert items["A"]["current_parent_id"] == "R"
    assert items["A"]["current_root_id"] == "R"
    # id_reused 优先于父/根变化判定：A 的 original_parent_id 恰好也是 R，
    # 即便父相同仍归类为 id_reused
    assert items["A"]["original_parent_id"] == "R"
    # C 未被重新归档、仍属 batch1，清理后提升为根 -> reparented
    assert items["C"]["status"] == "reparented"
    assert items["C"]["current_parent_id"] is None
    assert items["C"]["current_root_id"] == "C"
    assert audit["summary"] == {
        "total": 3, "deleted": 0, "id_reused": 2,
        "reparented": 1, "root_changed": 0, "unchanged": 0,
    }

    # 以新批次快照为基线：R/A 在本批内、拓扑一致 -> unchanged
    audit2 = client.get(f"/batches/{batch2['batch_id']}/drift").json()
    assert {it["message_id"]: it["status"] for it in audit2["results"]} == {
        "R": "unchanged", "A": "unchanged",
    }
    assert audit2["summary"] == {
        "total": 2, "deleted": 0, "id_reused": 0,
        "reparented": 0, "root_changed": 0, "unchanged": 2,
    }


def test_audit_only_covers_threads_nodes_not_cycle_members(client):
    # R 有效；x/y 成环 invalid（不进 threads，不被审计）
    batch = client.post("/batches", json={"messages": [
        _m("R", sent="2026-09-18T00:00:00+00:00"),
        _m("x", irt="y", refs=["y"], sent="2026-09-18T01:00:00+00:00"),
        _m("y", irt="x", refs=["x"], sent="2026-09-18T02:00:00+00:00"),
    ]}).json()
    audit = client.get(f"/batches/{batch['batch_id']}/drift").json()
    assert [it["message_id"] for it in audit["results"]] == ["R"]
    assert audit["summary"]["total"] == 1
    assert audit["summary"]["unchanged"] == 1


def test_drift_audit_is_read_only_and_does_not_rewrite_snapshot(client):
    # R(到期) -> A(到期) -> C：清理后再审计，快照必须保持归档时原样
    batch = client.post("/batches", json={"messages": [
        _m("R", sent="2026-09-18T08:00:00+00:00",
           retain_until="2026-09-19T00:00:00+00:00"),
        _m("A", irt="R", refs=["R"], sent="2026-09-18T09:00:00+00:00",
           retain_until="2026-09-19T00:00:00+00:00"),
        _m("C", irt="A", refs=["R", "A"], sent="2026-09-18T10:00:00+00:00"),
    ]}).json()
    snapshot_before = client.get(f"/batches/{batch['batch_id']}").json()

    client.post("/retention/cleanups", json={"cutoff": "2026-09-20T00:00:00+00:00"})
    c_after_cleanup = client.get("/messages/C").json()
    assert c_after_cleanup["parent_id"] is None

    audit1 = client.get(f"/batches/{batch['batch_id']}/drift").json()
    audit2 = client.get(f"/batches/{batch['batch_id']}/drift").json()
    # 只读：重复审计结论完全一致
    assert audit1 == audit2
    assert audit1["summary"] == {
        "total": 3, "deleted": 2, "id_reused": 0,
        "reparented": 1, "root_changed": 0, "unchanged": 0,
    }

    # 审计不改写当前拓扑：C 仍为压缩后的新根
    assert client.get("/messages/C").json() == c_after_cleanup
    # 快照原样返回（仍含已删除的 R/A）
    assert client.get(f"/batches/{batch['batch_id']}").json() == snapshot_before
    assert {t["message_id"] for t in snapshot_before["threads"]} == {"R"}
    assert client.get("/messages/R").status_code == 404
    assert client.get("/messages/A").status_code == 404
