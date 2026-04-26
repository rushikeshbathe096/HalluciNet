#!/usr/bin/env bash

echo "========================================"
echo "HALLUCINET — FINAL SUBMISSION VALIDATOR"
echo "========================================"

SPACE_URL="https://rushikeshbathe096-hallucinet.hf.space"
PASS=0
FAIL=0

check() {
    if [ $1 -eq 0 ]; then
        echo "✅ $2"
        PASS=$((PASS + 1))
    else
        echo "❌ $2"
        FAIL=$((FAIL + 1))
    fi
}

# ─── 1. HF SPACE LIVE ───────────────────────────────────────
echo ""
echo "=== 1. HF SPACE HEALTH ==="
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $SPACE_URL/health)
[ "$STATUS" = "200" ]
check $? "HF Space /health returns 200"

HEALTH=$(curl -s $SPACE_URL/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
[ "$HEALTH" = "healthy" ]
check $? "HF Space status=healthy (got: $HEALTH)"

# ─── 2. ENVIRONMENT ENDPOINTS ────────────────────────────────
echo ""
echo "=== 2. ENVIRONMENT ENDPOINTS ==="

for task in easy medium hard expert adversarial; do
    RESULT=$(curl -s -X POST $SPACE_URL/reset \
        -H "Content-Type: application/json" \
        -d "{\"task_id\": \"$task\"}" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); obs=d.get('observation',d.get('observation',d)); print('ok' if 'reference_document' in obs else 'fail')" 2>/dev/null)
    [ "$RESULT" = "ok" ]
    check $? "/reset task=$task returns reference_document"
done

# Step endpoint
STEP=$(curl -s -X POST $SPACE_URL/step \
    -H "Content-Type: application/json" \
    -d '{"action": {"has_hallucination": false, "hallucinated_claim": null, "correct_fact": null, "confidence": 0.5}}' | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('ok' if 'observation' in d or 'reward' in d else 'fail')" 2>/dev/null)
[ "$STEP" = "ok" ]
check $? "/step returns valid response"

# Generator endpoints
GEN=$(curl -s -X POST $SPACE_URL/generator/reset \
    -H "Content-Type: application/json" \
    -d '{"task_id": "easy"}' | python3 -c \
    "import sys,json; d=json.load(sys.stdin); obs=d.get('observation',d); print('ok' if 'task_id' in obs else 'fail')" 2>/dev/null)
[ "$GEN" = "ok" ]
check $? "/generator/reset returns valid response"

# ─── 3. NEW ENDPOINTS ────────────────────────────────────────
echo ""
echo "=== 3. NEW ENDPOINTS ==="

OVERSIGHT=$(curl -s $SPACE_URL/oversight | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('ok' if 'reliability_score' in d else 'fail')" 2>/dev/null)
[ "$OVERSIGHT" = "ok" ]
check $? "/oversight returns reliability_score"

DEBATE=$(curl -s -X POST $SPACE_URL/debate \
    -H "Content-Type: application/json" \
    -d '{"generator_defense": "test defense"}' | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('ok' if d.get('debate_round')==True else 'fail')" 2>/dev/null)
[ "$DEBATE" = "ok" ]
check $? "/debate returns debate_round=True"

LB=$(curl -s $SPACE_URL/leaderboard | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('ok' if len(d.get('leaderboard',[]))>0 else 'fail')" 2>/dev/null)
[ "$LB" = "ok" ]
check $? "/leaderboard returns entries"

# ─── 4. OPENENV VALIDATE ─────────────────────────────────────
echo ""
echo "=== 4. OPENENV VALIDATE ==="

VALIDATE=$(openenv validate --url $SPACE_URL 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d['summary']['passed_count'],'/',d['summary']['total_count'], 'passed' if d['passed'] else 'FAILED')" 2>/dev/null)
echo "Result: $VALIDATE"
[[ "$VALIDATE" == *"6 / 6"* ]]
check $? "openenv validate 6/6"

# ─── 5. DOCKER BUILD ─────────────────────────────────────────
echo ""
echo "=== 5. DOCKER BUILD ==="

if command -v docker &>/dev/null; then
    docker build . -t hallucinet-final-test -q 2>/dev/null
    check $? "Docker build succeeds"
else
    echo "⚠️  Docker not available — skipping"
fi

# ─── 6. LOCAL IMPORTS ────────────────────────────────────────
echo ""
echo "=== 6. LOCAL IMPORTS ==="

python3 -c "
from tasks import TASKS, list_tasks
from grader import grade
from curriculum import AdversarialCurriculumManager, TASK_ORDER
from models import HallucinationAction, GeneratorAction
from oversight_agent import OversightAgent
import inference
assert hasattr(inference, 'call_with_retry'), 'retry helper missing'
print('ok')
" 2>/dev/null
check $? "All local imports OK"

# ─── 7. TASK COUNTS ──────────────────────────────────────────
echo ""
echo "=== 7. TASK COUNTS ==="

python3 -c "
from tasks import TASKS
levels = list(TASKS.keys())
assert 'adversarial' in levels, 'adversarial task missing'
assert len(TASKS['easy']) >= 8, 'easy needs 8+ samples'
assert len(TASKS['expert']) >= 20, 'expert needs 20+ samples'
print('ok')
" 2>/dev/null
check $? "All task levels present with correct sample counts"

# ─── 8. GRADER SELF-TEST ─────────────────────────────────────
echo ""
echo "=== 8. GRADER ==="

python3 grader.py 2>/dev/null | grep -q "All 10 grader tests passed"
check $? "All 10 grader self-tests pass"

# ─── 9. SECRET CHECK ─────────────────────────────────────────
echo ""
echo "=== 9. SECRET CHECK ==="


! git ls-files | grep -q "^\.env$"
check $? ".env not committed to git"

grep -q "^\.env" .gitignore 2>/dev/null
check $? ".env in .gitignore"

# ─── 10. README LINKS ────────────────────────────────────────
echo ""
echo "=== 10. README LINKS ==="

grep -q "rushikeshbathe096-hallucinet.hf.space" README.md 2>/dev/null
check $? "README has correct HF Space URL"

grep -q "rushikeshbathe096/HalluciNet" README.md 2>/dev/null
check $? "README has GitHub URL"

# ─── SUMMARY ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo "SUMMARY: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
    echo "🎉 ALL CHECKS PASSED — READY TO SUBMIT"
else
    echo "⚠️  $FAIL checks failed — fix before submitting"
fi
echo "========================================"
