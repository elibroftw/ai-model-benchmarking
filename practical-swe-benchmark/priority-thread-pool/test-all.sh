#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# == 0 )); then
  set -- rust csharp typescript python cpp kotlin go
fi
for lang in "$@"; do
  case "$lang" in
    rust|csharp|typescript|python|cpp|kotlin|go) ;;
    *) echo "Unknown language: $lang" >&2; exit 2 ;;
  esac
  echo "== $lang =="
  (
    cd "$root/$lang-ptp"
    case "$lang" in
      rust) cargo test ;;
      csharp) dotnet run ;;
      typescript) npm ci --ignore-scripts; npm test ;;
      python) python3 -m unittest -v ;;
      cpp)
        cmake -S . -B build
        cmake --build build
        ./build/ptp_test
        ;;
      kotlin) ./gradlew test ;;
      go) go test -race -timeout 30s ./... ;;
    esac
  )
done
