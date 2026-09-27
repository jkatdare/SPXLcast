from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())     # the exit code matters: infra/daily.sh keeps the last good output on failure
