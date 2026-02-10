# th3cl4w Heartbeat Checks

## Service Health
Check services via systemd:
```bash
systemctl --user is-active th3cl4w-main.service
scripts/th3cl4w-ctl.sh status
```

If any service is not `active`, restart it:
```bash
scripts/th3cl4w-ctl.sh restart
```
