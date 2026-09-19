"""会话归并的纯逻辑。

父节点确定规则：
    1. 优先采用有效 in_reply_to（指向已知邮件，且不指向自身）；
    2. 否则从右向左扫描 references，取其中最后一个“已知项”；
    3. 都无法命中则自成根节点（断链邮件）。

引用环处理：
    候选父子关系构成一张“功能图”（每封新邮件至多一个候选父节点）。
    检测到引用环时，环上所有邮件整体标记为 invalid，绝不下沉、
    绝不部分挂接；随后让依赖该环的邮件重新在其引用中寻找其它有效
    父节点，重复检测直至没有环。指向历史库中已归档邮件的边不可能
    成环（历史邮件的父指针不可变）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional


@dataclass(frozen=True)
class ThreadMessage:
    """参与会话归并的一封邮件。"""

    message_id: str
    in_reply_to: Optional[str]
    references: tuple[str, ...]
    sent_at: str          # 统一为 UTC ISO 8601 字符串，便于稳定排序
    subject: str


@dataclass(frozen=True)
class Duplicate:
    """重复 message_id 诊断。"""

    message_id: str
    first_submitted_seq: int
    occurrences: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MissingRefs:
    """缺失引用诊断：列出该邮件引用中、已知集合里不存在的去重 ID。"""

    message_id: str
    missing: list[str]


@dataclass
class BatchResolution:
    """一批邮件归并后的完整结论。

    parent_of 仅包含最终有效挂接；环上邮件被整体剔除（invalid），
    不出现在 parent_of 中，也不挂到任何根节点下。
    """

    first_seen: dict[str, ThreadMessage]
    duplicates: list[Duplicate]
    missing_refs: list[MissingRefs]
    invalid_ids: set[str]
    cycles: list[list[str]]
    parent_of: dict[str, Optional[str]]
    roots: list[str]


def pick_candidate(
    msg: ThreadMessage,
    known_ids: set[str],
    invalid_ids: set[str],
) -> Optional[str]:
    """按优先级挑选候选父节点。

    - in_reply_to 有效（已知、非当前 invalid 环成员）则采用；
    - 否则取 references 中从右往左第一个满足条件的已知项；
    - 否则为 None（断链，自成根）。

    指向自身的引用也作为候选保留，从而形成自环，交由环检测整体判 invalid。
    """

    irt = msg.in_reply_to
    if (
        irt is not None
        and irt in known_ids
        and irt not in invalid_ids
    ):
        return irt

    for ref in reversed(msg.references):
        if ref in known_ids and ref not in invalid_ids:
            return ref
    return None


def find_cycles(
    parent_of: dict[str, Optional[str]],
) -> list[list[str]]:
    """在一张功能图（节点 -> 候选父节点或 None）上找出全部环。

    父节点为 None、或父 ID 不在映射中的边会穿出图，不可能成环。
    返回的每个环统一从环内最小 ID 起头，整体按环序列表排序。

    归并批次时 parent_of 只含本批新邮件，该函数即等价于在“新邮件
    子图”内找环；relink 修复时传入的是合并了既有父指针与全部候选
    关系的整图，从而跨候选统一检测环。
    """

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {mid: WHITE for mid in parent_of}
    cycles: list[list[str]] = []

    for start in sorted(parent_of):
        if color[start] != WHITE:
            continue
        stack: list[str] = []
        cur: Optional[str] = start
        while cur is not None and color.get(cur, BLACK) == WHITE:
            color[cur] = GRAY
            stack.append(cur)
            cur = parent_of.get(cur)
        else:
            # 以 GRAY 节点收尾 => 发现环
            if cur is not None and color.get(cur) == GRAY:
                cyc = stack[stack.index(cur):]
                k = cyc.index(min(cyc))
                cyc = cyc[k:] + cyc[:k]
                cycles.append(cyc)

        for node in stack:
            color[node] = BLACK

    cycles.sort(key=lambda c: c)
    return cycles


def resolve_batch(
    submitted: Iterable[ThreadMessage],
    existing_ids: Iterable[str] = (),
) -> BatchResolution:
    """归并一批邮件。

    Args:
        submitted: 调用方按顺序提交的邮件（同批可含重复 ID）。
        existing_ids: 历史批次中已归档的 message_id。

    Returns:
        BatchResolution，所有列表均已稳定排序。
    """

    submitted_list = list(submitted)

    # ---- 1. 同批去重，记录重复诊断 -------------------------------------
    first_index: dict[str, int] = {}
    first_seen: dict[str, ThreadMessage] = {}
    occurrences: dict[str, list[int]] = {}
    for idx, msg in enumerate(submitted_list, start=1):
        mid = msg.message_id
        if mid not in first_seen:
            first_index[mid] = idx
            first_seen[mid] = msg
            occurrences[mid] = [idx]
        else:
            occurrences[mid].append(idx)

    duplicates = [
        Duplicate(
            message_id=mid,
            first_submitted_seq=first_index[mid],
            occurrences=occurrences[mid],
        )
        for mid, occ in occurrences.items()
        if len(occ) > 1
    ]
    duplicates.sort(key=lambda d: (d.first_submitted_seq, d.message_id))

    new_ids = set(first_seen)
    known_ids = set(existing_ids) | new_ids

    # ---- 2. 缺失引用诊断（一次算定，与最终是否成环无关） ----------------
    missing_refs: list[MissingRefs] = []
    for mid in sorted(new_ids):
        msg = first_seen[mid]
        seen: set[str] = set()
        missing: list[str] = []
        for ref in msg.references:
            if ref not in known_ids and ref not in seen:
                seen.add(ref)
                missing.append(ref)
        irt = msg.in_reply_to
        if irt is not None and irt not in known_ids and irt not in seen:
            seen.add(irt)
            missing.append(irt)
        if missing:
            missing.sort()
            missing_refs.append(MissingRefs(message_id=mid, missing=missing))
    missing_refs.sort(key=lambda m: m.message_id)

    # ---- 3. 候选父节点 + 定点评测：环整体 invalid，绝不部分挂接 ---------
    invalid_ids: set[str] = set()
    cycles: list[list[str]] = []
    while True:
        parent_of: dict[str, Optional[str]] = {}
        for mid, msg in first_seen.items():
            if mid in invalid_ids:
                continue
            parent_of[mid] = pick_candidate(msg, known_ids, invalid_ids)

        # 父指针为历史邮件或为 None 的边会穿出本批子图，不可能成环；
        # find_cycles 对不在映射中的父节点天然视为穿出。
        found = find_cycles(parent_of)
        if not found:
            break

        for cyc in found:
            invalid_ids.update(cyc)
        cycles.extend(found)

    roots = sorted(
        (mid for mid, pid in parent_of.items() if pid is None),
        key=lambda m: (first_seen[m].sent_at, m),
    )

    return BatchResolution(
        first_seen=first_seen,
        duplicates=duplicates,
        missing_refs=missing_refs,
        invalid_ids=invalid_ids,
        cycles=cycles,
        parent_of=parent_of,
        roots=roots,
    )


def _sort_key(sent_at: str, message_id: str) -> tuple[str, str]:
    return (sent_at, message_id)


def _sorted_tree(
    messages: dict[str, ThreadMessage],
    parent_of: dict[str, Optional[str]],
) -> tuple[list[str], dict[str, list[str]]]:
    """构造根列表与子节点表，并按 (sent_at, message_id) 稳定排序。

    嵌套树（build_forest）与扁平先序展开（build_flat_tree）共用同一套
    排序结果，保证两种展开的节点顺序完全一致。
    """

    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for mid, pid in parent_of.items():
        if pid is None:
            roots.append(mid)
        else:
            children.setdefault(pid, []).append(mid)

    def node_key(mid: str) -> tuple[str, str]:
        return _sort_key(messages[mid].sent_at, mid)

    roots.sort(key=node_key)
    for kids in children.values():
        kids.sort(key=node_key)
    return roots, children


def build_forest(
    messages: dict[str, ThreadMessage],
    parent_of: dict[str, Optional[str]],
) -> list[dict]:
    """把 parent_of 展开为稳定排序的会话树。

    每棵树按 (sent_at, message_id) 排序；同一父节点下的子节点、
    以及根之间都使用同样的键，保证全序稳定。
    """

    roots, children = _sorted_tree(messages, parent_of)

    def to_node(mid: str) -> dict:
        msg = messages[mid]
        return {
            "message_id": msg.message_id,
            "parent_id": parent_of.get(mid),
            "in_reply_to": msg.in_reply_to,
            "references": list(msg.references),
            "sent_at": msg.sent_at,
            "subject": msg.subject,
            "children": [to_node(c) for c in children.get(mid, [])],
        }

    return [to_node(r) for r in roots]


def build_flat_tree(
    messages: dict[str, ThreadMessage],
    parent_of: dict[str, Optional[str]],
) -> list[dict]:
    """与 build_forest 相同的排序，把会话树先序展开为扁平节点列表。

    每个节点在树节点字段的基础上增加 depth（根为 0），不含 children；
    供 GET /threads/{message_id}/page 做偏移分页。
    """

    roots, children = _sorted_tree(messages, parent_of)

    def to_item(mid: str, depth: int) -> list[dict]:
        msg = messages[mid]
        head = {
            "message_id": msg.message_id,
            "parent_id": parent_of.get(mid),
            "in_reply_to": msg.in_reply_to,
            "references": list(msg.references),
            "sent_at": msg.sent_at,
            "subject": msg.subject,
            "depth": depth,
        }
        out = [head]
        for kid in children.get(mid, []):
            out.extend(to_item(kid, depth + 1))
        return out

    items: list[dict] = []
    for root in roots:
        items.extend(to_item(root, 0))
    return items


# ---------------------------------------------------------------------------
# 保留期清理：到期删除 + 会话压缩（纯逻辑，dry_run 与实际清理共用）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanupImpact:
    """一个存活节点在压缩前后的父节点 / 根节点。"""

    message_id: str
    old_parent_id: Optional[str]
    new_parent_id: Optional[str]
    old_root_id: str
    new_root_id: str


@dataclass(frozen=True)
class RetentionCleanupPlan:
    """一次保留期清理的完整结论。

    deleted_ids 与 impacts 均已按 message_id 稳定排序；expired_ids 为
    全部到期有效邮件，protected_expired_ids 为其中“到期且被冻结保护”
    的子集，二者之差即实际删除集。
    """

    deleted_ids: list[str]
    impacts: list[CleanupImpact]
    expired_ids: frozenset[str]
    protected_expired_ids: frozenset[str]


def plan_retention_cleanup(
    parent_of: dict[str, Optional[str]],
    expired_ids: set[str],
    protected_ids: set[str],
) -> RetentionCleanupPlan:
    """在当前有效邮件森林上计算到期清理与压缩结果。

    - 删除集 = 到期邮件中未被冻结保护者（删除集之外的节点全部存活）；
    - 存活节点若父节点被删，则沿父链向上跳过全部被删节点，重挂到
      最近的存活祖先；链顶没有存活祖先时成为新根；
    - 根节点不变的判定基于压缩后的父指针：祖先链中间被删只改父指针，
      根节点被删则整棵剩余子树的根随之改变。

    入参 parent_of 只包含有效邮件（环内 invalid 邮件从不参与），
    expired_ids 中不属于该森林的 ID 会被忽略。
    """

    valid_ids = set(parent_of)
    expired_valid = {mid for mid in expired_ids if mid in valid_ids}
    protected_expired = expired_valid & set(protected_ids)
    delete_set = expired_valid - set(protected_ids)
    if not delete_set:
        return RetentionCleanupPlan(
            deleted_ids=[],
            impacts=[],
            expired_ids=frozenset(expired_valid),
            protected_expired_ids=frozenset(protected_expired),
        )

    survivors = valid_ids - delete_set

    # 旧根（压缩前）：沿父指针上溯
    def old_root_of(mid: str) -> str:
        cur = mid
        seen: set[str] = set()
        while True:
            pid = parent_of.get(cur)
            if pid is None or pid not in valid_ids or pid in seen:
                return cur
            seen.add(cur)
            cur = pid

    old_root: dict[str, str] = {mid: old_root_of(mid) for mid in survivors}

    # 新父：沿父链跳过被删节点，找最近存活祖先；无则成根（None）。
    # 记忆化：最近存活祖先只取决于父链上更靠上的节点，先把父链节点
    # 收集起来，再自顶向下回填，保证每个节点只走一遍链。
    new_parent: dict[str, Optional[str]] = {}

    def resolve(mid: str) -> Optional[str]:
        if mid in new_parent:
            return new_parent[mid]
        chain: list[str] = []
        cur = mid
        seen: set[str] = set()
        # 收集父链直到根、穿出森林、回到已解析节点（防御性：遇环也停）
        while (
            cur is not None
            and cur in valid_ids
            and cur not in new_parent
            and cur not in seen
        ):
            chain.append(cur)
            seen.add(cur)
            cur = parent_of.get(cur)
        inherited = new_parent.get(cur) if cur is not None else None
        for node in reversed(chain):
            pid = parent_of.get(node)
            if pid is None or pid not in valid_ids:
                inherited = None
            elif pid not in delete_set:
                inherited = pid
            # pid 被删 => inherited 保持为已解析出的最近存活祖先
            new_parent[node] = inherited
        return new_parent[mid]

    for mid in survivors:
        resolve(mid)

    # 新根（压缩后）：在新父指针上一次上溯
    def new_root_of(mid: str) -> str:
        cur = mid
        seen: set[str] = set()
        while True:
            pid = new_parent.get(cur)
            if pid is None or pid in seen:
                return cur
            seen.add(cur)
            cur = pid

    new_root: dict[str, str] = {mid: new_root_of(mid) for mid in survivors}

    impacts: list[CleanupImpact] = []
    for mid in survivors:
        op, nr = new_parent[mid], new_root[mid]
        if op != parent_of.get(mid) or nr != old_root[mid]:
            impacts.append(CleanupImpact(
                message_id=mid,
                old_parent_id=parent_of.get(mid),
                new_parent_id=op,
                old_root_id=old_root[mid],
                new_root_id=nr,
            ))
    impacts.sort(key=lambda i: i.message_id)

    return RetentionCleanupPlan(
        deleted_ids=sorted(delete_set),
        impacts=impacts,
        expired_ids=frozenset(expired_valid),
        protected_expired_ids=frozenset(protected_expired),
    )


# ---------------------------------------------------------------------------
# 会话响应时限（SLA）评估：纯逻辑，只读
# ---------------------------------------------------------------------------

# 逐封判定的四种结论。
SLA_WITHIN = "within_sla"   # 已有有效响应，且响应耗时不超过阈值
SLA_LATE = "late"           # 已有有效响应，但响应耗时超过阈值
SLA_PENDING = "pending"     # 尚无有效响应，以 as_of 计算的年龄未超阈值
SLA_OVERDUE = "overdue"     # 尚无有效响应，以 as_of 计算的年龄已超阈值


@dataclass(frozen=True)
class SlaItem:
    """一封邮件的响应时限判定。

    response_* 仅当存在有效响应（最早合格直接子邮件）时非空；
    age_seconds 仅当无有效响应时非空（以 as_of 计算的年龄）。
    """

    message_id: str
    parent_id: Optional[str]
    sent_at: str
    depth: int
    status: str
    response_message_id: Optional[str]
    response_sent_at: Optional[str]
    response_delay_seconds: Optional[float]
    age_seconds: Optional[float]


@dataclass(frozen=True)
class ChronologyAnomaly:
    """sent_at 早于父邮件的直接子邮件：时序异常，不得作为响应。"""

    message_id: str
    parent_id: str
    sent_at: str
    parent_sent_at: str


@dataclass(frozen=True)
class ThreadSlaEvaluation:
    """整棵会话的响应时限评估结论（items 按稳定先序排列）。"""

    items: list[SlaItem]
    anomalies: list[ChronologyAnomaly]
    summary: dict[str, int]


def _parse_sent_at(iso: str) -> datetime:
    """把库里的 UTC ISO 8601 字符串解析为 aware datetime（naive 视为 UTC）。"""

    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def evaluate_thread_sla(
    messages: dict[str, ThreadMessage],
    parent_of: dict[str, Optional[str]],
    as_of: datetime,
    threshold_seconds: float,
) -> ThreadSlaEvaluation:
    """对整棵会话逐封判定响应时限（纯函数，不改动任何输入）。

    - 有效响应：直接子邮件中 sent_at 不早于该邮件 sent_at 的最早一封
      （子节点已按 (sent_at, message_id) 稳定排序，首个合格者即最早者）；
    - sent_at 早于父邮件的直接子邮件记入 anomalies，不得消除超时；
    - 无有效响应：年龄 = as_of - sent_at，超过阈值 overdue，否则 pending；
    - 已有有效响应：耗时不超过阈值 within_sla，否则 late（边界含等于）。
    """

    sent = {mid: _parse_sent_at(m.sent_at) for mid, m in messages.items()}
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    as_of = as_of.astimezone(timezone.utc)

    roots, children = _sorted_tree(messages, parent_of)

    items: list[SlaItem] = []
    anomalies: list[ChronologyAnomaly] = []
    counts = {SLA_WITHIN: 0, SLA_LATE: 0, SLA_PENDING: 0, SLA_OVERDUE: 0}

    def visit(mid: str, depth: int) -> None:
        parent_sent = sent[mid]
        response_id: Optional[str] = None
        for kid in children.get(mid, []):
            if sent[kid] >= parent_sent:
                if response_id is None:
                    response_id = kid  # 首个合格者即最早者
            else:
                anomalies.append(ChronologyAnomaly(
                    message_id=kid,
                    parent_id=mid,
                    sent_at=messages[kid].sent_at,
                    parent_sent_at=messages[mid].sent_at,
                ))

        if response_id is not None:
            delay = (sent[response_id] - parent_sent).total_seconds()
            status = SLA_WITHIN if delay <= threshold_seconds else SLA_LATE
            items.append(SlaItem(
                message_id=mid,
                parent_id=parent_of.get(mid),
                sent_at=messages[mid].sent_at,
                depth=depth,
                status=status,
                response_message_id=response_id,
                response_sent_at=messages[response_id].sent_at,
                response_delay_seconds=delay,
                age_seconds=None,
            ))
        else:
            age = (as_of - parent_sent).total_seconds()
            status = SLA_OVERDUE if age > threshold_seconds else SLA_PENDING
            items.append(SlaItem(
                message_id=mid,
                parent_id=parent_of.get(mid),
                sent_at=messages[mid].sent_at,
                depth=depth,
                status=status,
                response_message_id=None,
                response_sent_at=None,
                response_delay_seconds=None,
                age_seconds=age,
            ))
        counts[status] += 1
        for kid in children.get(mid, []):
            visit(kid, depth + 1)

    for root in roots:
        visit(root, 0)

    summary = {"total": len(items), **counts}
    return ThreadSlaEvaluation(items=items, anomalies=anomalies, summary=summary)


# ---------------------------------------------------------------------------
# 会话主题漂移审计：纯逻辑，只读
# ---------------------------------------------------------------------------

# 大小写不敏感的 Re:/Fw:/Fwd: 前缀（须带冒号），重复剥离直至不再匹配。
_SUBJECT_PREFIX_RE = re.compile(r"^\s*(?:re|fw|fwd)\s*:\s*", re.IGNORECASE)


def normalize_subject(subject: str) -> str:
    """计算规范主题：重复剥离大小写不敏感的 Re/Fw/Fwd 前缀并折叠空白。

    前缀必须带冒号（Re: / Fw: / Fwd:），大小写不限，可连续叠加出现
    （"Re: Fw: x" 逐层剥离）；随后把任意连续空白折叠为单个空格并去除
    首尾空白。漂移判定按规范主题精确相等比较。
    """

    s = subject
    while True:
        stripped = _SUBJECT_PREFIX_RE.sub("", s, count=1)
        if stripped == s:
            break
        s = stripped
    return " ".join(s.split())


@dataclass(frozen=True)
class SubjectAuditItem:
    """一封邮件的主题漂移判定。

    baseline_subject 为分支基线（最近祖先——即父节点——的规范主题）；
    根节点无祖先，baseline 为 None 且恒不偏离。
    """

    message_id: str
    parent_id: Optional[str]
    depth: int
    subject: str
    canonical_subject: str
    baseline_subject: Optional[str]
    drifted: bool


@dataclass(frozen=True)
class SubjectDriftGroup:
    """先序序列中连续的一段偏离节点；baseline_subject 为组首节点偏离的基线。"""

    members: list[str]
    baseline_subject: str


@dataclass(frozen=True)
class ThreadSubjectAudit:
    """整棵会话的主题漂移审计结论（items 按稳定先序排列）。"""

    items: list[SubjectAuditItem]
    drift_groups: list[SubjectDriftGroup]


def audit_thread_subjects(
    messages: dict[str, ThreadMessage],
    parent_of: dict[str, Optional[str]],
) -> ThreadSubjectAudit:
    """对整棵会话逐节点审计主题漂移（纯函数，不改动任何输入）。

    - 按稳定先序遍历（与 build_flat_tree 同一套排序）；
    - 每个节点的分支基线 = 其最近祖先（父节点）的规范主题；根节点无基线、
      恒不偏离（它定义整棵树的初始基线）；
    - 父节点自身偏离时，子节点以父节点偏离后的新主题为基线——分支基线
      随谱系逐层更新，沿用新话题的分支不再视为漂移；
    - 节点规范主题与基线不同即偏离（精确比较）；
    - drift_groups 把先序序列中连续偏离的节点合并为一组，组首节点所
      偏离的基线记为该组的 baseline_subject。
    """

    roots, children = _sorted_tree(messages, parent_of)

    items: list[SubjectAuditItem] = []
    drift_groups: list[SubjectDriftGroup] = []
    group_members: list[str] = []
    group_baseline: Optional[str] = None

    def flush_group() -> None:
        nonlocal group_members, group_baseline
        if group_members:
            # 偏离节点必有基线（根节点恒不偏离，不会进入分组）
            drift_groups.append(SubjectDriftGroup(
                members=group_members,
                baseline_subject=group_baseline if group_baseline is not None else "",
            ))
        group_members = []
        group_baseline = None

    def visit(mid: str, depth: int, baseline: Optional[str]) -> None:
        nonlocal group_members, group_baseline
        msg = messages[mid]
        canonical = normalize_subject(msg.subject)
        drifted = baseline is not None and canonical != baseline
        items.append(SubjectAuditItem(
            message_id=mid,
            parent_id=parent_of.get(mid),
            depth=depth,
            subject=msg.subject,
            canonical_subject=canonical,
            baseline_subject=baseline,
            drifted=drifted,
        ))
        if drifted:
            if not group_members:
                group_baseline = baseline
            group_members.append(mid)
        else:
            flush_group()
        for kid in children.get(mid, []):
            visit(kid, depth + 1, canonical)

    for root in roots:
        visit(root, 0, None)
    flush_group()

    return ThreadSubjectAudit(items=items, drift_groups=drift_groups)


# ---------------------------------------------------------------------------
# 批次漂移审计：以批次不可变结果快照中的原始会话树为基线，只读比对当前拓扑
# ---------------------------------------------------------------------------

# 逐节点漂移状态（判定优先级自上而下）。
DRIFT_DELETED = "deleted"          # 同 ID 当前已不存在（如已被保留期清理删除）
DRIFT_ID_REUSED = "id_reused"      # 同 ID 当前存在，但属于另一个批次（清理后同 ID 重归档）
DRIFT_REPARENTED = "reparented"    # 来源批次一致，但当前父节点与原始父节点不同
DRIFT_ROOT_CHANGED = "root_changed"  # 父节点未变，但所在会话的根已变化
DRIFT_UNCHANGED = "unchanged"      # 父节点与根均与快照一致


@dataclass(frozen=True)
class BatchDriftItem:
    """快照 threads 中一个原始节点相对当前拓扑的漂移判定。"""

    message_id: str
    original_parent_id: Optional[str]
    current_parent_id: Optional[str]
    original_root_id: str
    current_root_id: Optional[str]
    status: str


@dataclass(frozen=True)
class BatchDriftAudit:
    """一次批次漂移审计的完整结论（items 按快照原始先序排列）。"""

    items: list[BatchDriftItem]
    summary: dict[str, int]


def flatten_forest_preorder(
    threads: list[dict],
) -> list[tuple[str, Optional[str]]]:
    """把快照中嵌套的会话树按**原始先序**展开为 (message_id, parent_id) 序列。

    快照 threads 由 build_forest 以 (sent_at, message_id) 稳定排序生成，
    嵌套 children 的次序即先序；此处仅做保持次序的扁平化，不重新排序，
    审计因此严格按归档时的原始先序逐封输出。
    """

    out: list[tuple[str, Optional[str]]] = []

    def walk(node: dict) -> None:
        out.append((node["message_id"], node.get("parent_id")))
        for child in node.get("children", []):
            walk(child)

    for root in threads:
        walk(root)
    return out


def _climb_in_parent_map(
    start: str, parent_of: dict[str, Optional[str]]
) -> str:
    """在内存父指针图上向上爬到根（遇环或穿出时回退为当前节点）。"""

    cur = start
    seen: set[str] = set()
    while True:
        pid = parent_of.get(cur)
        if pid is None or pid in seen:
            return cur
        seen.add(cur)
        cur = pid


def audit_batch_drift(
    batch_id: str,
    original_nodes: list[tuple[str, Optional[str]]],
    current_rows: dict[str, tuple[str, Optional[str], bool]],
) -> BatchDriftAudit:
    """以批次快照中的原始会话树为基线，逐节点比对当前拓扑（纯函数）。

    Args:
        batch_id: 被审计批次的 ID（基线来源）。
        original_nodes: 快照 threads 按原始先序展开的
            (message_id, original_parent_id) 序列；仅审计这些节点。
        current_rows: 当前 messages 表中 message_id ->
            (batch_id, parent_id, is_invalid) 的映射。

    状态按以下优先级**依次**判定（命中即止）：

    1. ``deleted``：当前 messages 中不存在同 ID；
    2. ``id_reused``：同 ID 当前存在，但 ``batch_id`` 不是被审计批次
       （典型为清理删除后同 ID 在新批次重新归档）；
    3. ``reparented``：来源批次一致，但当前父节点与原始父节点不同；
    4. ``root_changed``：父节点未变，但当前所在会话的根与原始根不同；
    5. ``unchanged``：其余情况。

    环内 invalid 邮件从不出现在快照 threads 中；重归档后的同 ID 若落入
    引用环（invalid），其 current_parent_id/current_root_id 输出 None。
    """

    original_parent_of: dict[str, Optional[str]] = {
        mid: pid for mid, pid in original_nodes
    }
    current_parent_of: dict[str, Optional[str]] = {
        mid: row[1] for mid, row in current_rows.items()
    }

    counts = {
        DRIFT_DELETED: 0,
        DRIFT_ID_REUSED: 0,
        DRIFT_REPARENTED: 0,
        DRIFT_ROOT_CHANGED: 0,
        DRIFT_UNCHANGED: 0,
    }
    items: list[BatchDriftItem] = []

    for mid, original_pid in original_nodes:
        original_root = _climb_in_parent_map(mid, original_parent_of)

        row = current_rows.get(mid)
        if row is None:
            # 1) 当前不存在 -> deleted
            status = DRIFT_DELETED
            current_pid: Optional[str] = None
            current_root: Optional[str] = None
        else:
            current_batch, row_parent_id, is_invalid = row
            if is_invalid:
                # 环内 invalid 邮件不挂接任何会话树
                current_pid = None
                current_root = None
            else:
                current_pid = row_parent_id
                current_root = _climb_in_parent_map(mid, current_parent_of)

            if current_batch != batch_id:
                # 2) 同 ID 当前属于其他批次 -> id_reused
                status = DRIFT_ID_REUSED
            elif current_pid != original_pid:
                # 3) 来源一致但父节点变化（补链、纠偏、清理压缩）-> reparented
                status = DRIFT_REPARENTED
            elif current_root != original_root:
                # 4) 仅根变化（祖先链上的根被删而本节点父指针未变）-> root_changed
                status = DRIFT_ROOT_CHANGED
            else:
                # 5) 父与根均未变 -> unchanged
                status = DRIFT_UNCHANGED

        counts[status] += 1
        items.append(BatchDriftItem(
            message_id=mid,
            original_parent_id=original_pid,
            current_parent_id=current_pid,
            original_root_id=original_root,
            current_root_id=current_root,
            status=status,
        ))

    summary = {"total": len(items), **counts}
    return BatchDriftAudit(items=items, summary=summary)
