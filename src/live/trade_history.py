"""Observational trade records; never used to make trading decisions."""

from datetime import datetime, timezone
import re


def recover_open_time(position, log_path):
    """Recover only an unambiguous current paper fill from the UTC service log.

    Old log prices are rounded, so both price and notional must match. Never
    search past a later close or conflicting entry, or use a state-file mtime.
    """
    if not position or position.get('opened_at') or position.get('live'):
        return position
    result = dict(position)
    try:
        with log_path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 2_000_000))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        for line in reversed(lines):
            if '平仓:' in line:
                break
            if '[纸上]' not in line:
                continue
            match = re.search(r'\[纸上\] (BUY|SELL) .*?@([\d.]+) 名义=([\d.]+)U', line)
            if not match:
                continue
            side, price, notional = match.groups()
            expected = 'BUY' if result['side'] == 'long' else 'SELL'
            if (side == expected and price == f"{result['entry']:.0f}"
                    and notional == f"{result['notional']:.1f}"):
                stamp = datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S.%f')
                result.update(opened_at=stamp.replace(tzinfo=timezone.utc).isoformat(),
                              opened_at_source='production_log_utc')
            break
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return result


def open_event(position):
    return {"kind": "strategy_open", "status": "filled",
            "time": position['opened_at'],
            **{key: position.get(key) for key in (
                'side', 'entry', 'size', 'notional', 'tp', 'sl',
                'opened_at_source', 'entry_analysis')}}
