Engine code remains hand-written. You are not allowed to change
engine production code unless the user explicitly requests it.
Even if you see obvious problems - you are not allowed to change it.
Even if tests are failing - still not allowed.

You are allowed to write tests and tooling for tests.
If tests are failing - escalate to the user. Do not fix
them until user says so.

Always run tests with 10s timeout:
`go test -timeout=10s`, there is forever
loop running in cordon extraction, which has breaking rules,
but I have no math proof it always finishes. I investigate
those cases as they arise.
