#!/bin/zsh
# Task 5 (D157, D160): every system on the 148 test volumes, once, with the frozen texts. Two lanes run side by side
# (they share the tool cache; a cache entry is written to a temporary folder and renamed, so they cannot collide):
#   caffeinate -i zsh scripts/test5.sh compute   ② chain, then ②u (2 compute workers)
#   caffeinate -i zsh scripts/test5.sh models    Haiku, luna, Qwen on ③ ③b ④a ④; Haiku's ④ twice more (D131);
#                                                then, after the compute lane is done too, Terra and Sonnet on ④ only
#                                                (the user's order); then the scoring with ⑤ (D151) and the cards
# → ../dataset/task4-results/stage5-test/ (and stage5-test-rep2, -rep3). The smoke test runs before each model system
# (its pass must be from the same day, D135; the run crosses midnight). A run that stops short is tried once more
# (finished cases are skipped); if it is still short the lane goes on and says INCOMPLETE.
# 30 Oct (D164): side by side the lanes held 3 workers of about 6.8 GB each on the 16 GB Mac (swap 17.5 of 18 GB),
# against the 8 GB rule. The compute lane was stopped after 18 volumes and restarts, with 1 worker
# (`test5.sh compute 1`), once the models lane is waiting for it: one worker at a time.
# 30 Oct (D165): the Wi-Fi went down ~09:45–11:30: 80 Qwen ④a cases set aside to <out>/_network/, Qwen ③ ③b and
# Haiku's ④ repeats not started (no smoke pass). The models lane was stopped and runs again in full once the
# compute lane is done (finished systems are skipped), then Terra, Sonnet, the scoring and the cards.
# D166 (Wi-Fi off by choice): after the compute lane, ⑤ and a first scoring run offline (1 worker); the models
# lane starts when the network has answered 3 checks in a row.
# 31 Oct (D167): 11 Terra cases ended on OpenRouter's limit for new accounts (20 requests a minute on this model) with
# 6 cases at a time; they were set aside to <out>/_ratelimit/ and run again. Terra runs 1 case at a time (~13 a
# minute), Sonnet 2; the pace changes, not what the model sees.
cd "${0:A:h}/.."
OUT=stage5-test
RUN=(uv run python scripts/run_agent.py)
DONE=../dataset/task4-results/$OUT/.compute_done

model_run() {  # out model system [cases at a time, default 6]
  T=(--threads ${4:-6})
  $RUN smoke --model $2 --out $1 || $RUN smoke --model $2 --out $1 || { echo "SMOKE FAIL $2"; return 1 }
  $RUN run --system $3 --model $2 --cases test --out $1 --workers 1 $T \
    || $RUN run --system $3 --model $2 --cases test --out $1 --workers 1 $T || echo "INCOMPLETE $1 $3-$2"
}

case $1 in
  compute)  # [workers, default 2]
    W=(--workers ${2:-2})
    $RUN run --system chain --cases test --out $OUT $W || $RUN run --system chain --cases test --out $OUT $W
    $RUN run --system chain_u --cases test --out $OUT $W || $RUN run --system chain_u --cases test --out $OUT $W
    touch $DONE
    echo "COMPUTE LANE DONE" ;;
  models)
    for m in haiku luna qwen; do
      for s in agent agent_eyes router router_readings; do model_run $OUT $m $s; done
    done
    for r in 2 3; do model_run $OUT-rep$r haiku agent; done
    until [[ -f $DONE ]]; do sleep 120; done
    model_run $OUT terra agent 1  # D167
    model_run $OUT sonnet agent 2
    uv run python scripts/score_agent.py --out $OUT --oracle --workers 1  # D166
    for r in 2 3; do uv run python scripts/score_agent.py --out $OUT-rep$r; done
    for o in $OUT $OUT-rep2 $OUT-rep3; do uv run python scripts/paper_cards.py --out $o; done
    echo "TASK5 DONE" ;;
  *) echo "usage: test5.sh compute|models"; exit 2 ;;
esac
