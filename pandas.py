from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class Mask(list):
    def __invert__(self):
        return Mask([not bool(item) for item in self])

    def all(self) -> bool:
        return all(bool(x) for x in self)


class Index(list):
    def __init__(self, data: Iterable[Any], name: Optional[str] = None):
        super().__init__(data)
        self.name = name

    def duplicated(self, keep: str = "first") -> Mask:
        seen = set()
        duplicates: List[bool] = []
        if keep == "last":
            seen = set()
            for item in reversed(self):
                duplicates.insert(0, item in seen)
                seen.add(item)
            return Mask(duplicates)
        for item in self:
            duplicates.append(item in seen)
            seen.add(item)
        return Mask(duplicates)


class Series(list):
    def __init__(self, data: Iterable[Any]):
        super().__init__(data)

    def nunique(self) -> int:
        return len(set(self))

    def all(self) -> bool:
        return all(bool(x) for x in self)

    def __eq__(self, other):
        return Mask([item == other for item in self])


class DataFrame:
    def __init__(self, data: Any = None, index: Optional[Iterable[Any]] = None):
        self._data: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            keys = list(data.keys())
            values = list(data.values())
            length = len(values[0]) if values else 0
            for i in range(length):
                row = {k: values[idx][i] for idx, k in enumerate(keys)}
                self._data.append(row)
        elif isinstance(data, list):
            for row in data:
                self._data.append(dict(row))
        elif data is None:
            self._data = []
        else:
            raise TypeError("Unsupported data type for DataFrame")

        if index is None:
            self.index = Index(range(len(self._data)))
        else:
            self.index = Index(list(index))
        self.index.name = getattr(index, "name", None)

    @property
    def columns(self) -> List[str]:
        if not self._data:
            return []
        # preserve insertion order from first row
        return list(self._data[0].keys())

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def copy(self) -> "DataFrame":
        copied = DataFrame()
        copied._data = [dict(row) for row in self._data]
        copied.index = Index(self.index, name=self.index.name)
        return copied

    def iterrows(self):
        for idx, row in zip(self.index, self._data):
            yield idx, SimpleRow(row)

    def rename(self, *, columns: Dict[str, str], inplace: bool = False):
        target = self if inplace else self.copy()
        for row in target._data:
            for old, new in columns.items():
                if old in row:
                    row[new] = row.pop(old)
        # adjust columns order
        if not target.empty:
            target._data = [dict(row) for row in target._data]
        return None if inplace else target

    def set_index(self, keys, inplace: bool = False):
        target = self if inplace else self.copy()
        if isinstance(keys, list):
            new_index = [tuple(row.get(k) for k in keys) for row in target._data]
            for row in target._data:
                for k in keys:
                    row.pop(k, None)
            name = None
        else:
            new_index = [row.pop(keys, None) for row in target._data]
            name = keys
        target.index = Index(new_index)
        target.index.name = name
        return None if inplace else target

    def sort_index(self, inplace: bool = False):
        pairs = sorted(zip(self.index, self._data), key=lambda x: x[0])
        sorted_index = [p[0] for p in pairs]
        sorted_data = [p[1] for p in pairs]
        if inplace:
            self.index = Index(sorted_index, name=self.index.name)
            self._data = sorted_data
            return None
        new_df = self.copy()
        new_df.index = Index(sorted_index, name=self.index.name)
        new_df._data = [dict(row) for row in sorted_data]
        return new_df

    def __getitem__(self, key):
        if isinstance(key, list) and key and isinstance(key[0], bool):
            filtered_data = [row for row, flag in zip(self._data, key) if flag]
            filtered_index = [idx for idx, flag in zip(self.index, key) if flag]
            df = DataFrame(filtered_data)
            df.index = Index(filtered_index, name=self.index.name)
            return df
        if isinstance(key, list):
            rows = []
            for row in self._data:
                rows.append({k: row[k] for k in key})
            df = DataFrame(rows)
            df.index = Index(self.index, name=self.index.name)
            return df
        column = Series([row.get(key) for row in self._data])
        return column

    def __setitem__(self, key, value):
        if isinstance(key, list):
            if isinstance(value, DataFrame):
                for idx, row in enumerate(self._data):
                    source = value._data[idx]
                    for col in key:
                        row[col] = source.get(col)
                return
            if isinstance(value, list):
                for row, val in zip(self._data, value):
                    if isinstance(val, dict):
                        for col in key:
                            row[col] = val.get(col)
                    else:
                        row[key[0]] = val
                return
        if isinstance(value, list):
            for row, val in zip(self._data, value):
                row[key] = val
        else:
            for row in self._data:
                row[key] = value

    def dropna(self, subset: List[str], inplace: bool = False):
        filtered_data = []
        filtered_index = []
        for idx, row in zip(self.index, self._data):
            if all(row.get(col) is not None for col in subset):
                filtered_data.append(row)
                filtered_index.append(idx)
        if inplace:
            self._data = filtered_data
            self.index = Index(filtered_index, name=self.index.name)
            return None
        df = DataFrame(filtered_data)
        df.index = Index(filtered_index, name=self.index.name)
        return df

    def apply(self, func, errors: str = "raise"):
        applied_rows = []
        for row in self._data:
            new_row = {}
            for key, value in row.items():
                try:
                    new_row[key] = func(value)
                except Exception:
                    if errors == "coerce":
                        new_row[key] = None
                    else:
                        raise
            applied_rows.append(new_row)
        df = DataFrame(applied_rows)
        df.index = Index(self.index, name=self.index.name)
        return df

    def groupby(self, key: str):
        groups: Dict[Any, List[Tuple[Any, Dict[str, Any]]]] = {}
        for idx, row in zip(self.index, self._data):
            k = row.get(key)
            groups.setdefault(k, []).append((idx, row))
        return GroupBy(groups, key)

    def to_csv(self, path: Path, index: bool = False):
        lines = []
        cols = self.columns
        if index:
            cols = ["index"] + cols
        lines.append(",".join(cols))
        for idx, row in zip(self.index, self._data):
            values = [row.get(col, "") for col in self.columns]
            if index:
                values = [idx] + values
            lines.append(",".join(str(v) for v in values))
        Path(path).write_text("\n".join(lines))

    def reset_index(self):
        rows = []
        index_name = self.index.name or "index"
        for idx, row in zip(self.index, self._data):
            new_row = {index_name: idx}
            new_row.update(row)
            rows.append(new_row)
        df = DataFrame(rows)
        df.index = Index(range(len(rows)))
        return df

    def __len__(self):
        return len(self._data)


class GroupBy:
    def __init__(self, groups: Dict[Any, List[Tuple[Any, Dict[str, Any]]]], key: str):
        self.groups = groups
        self.key = key

    def __iter__(self):
        for key, rows in self.groups.items():
            df = DataFrame([dict(r[1]) for r in rows])
            df.index = Index([r[0] for r in rows])
            return_iter = (key, df)
            yield return_iter


def to_datetime(values):
    if isinstance(values, list):
        return [datetime.fromisoformat(str(v)) for v in values]
    return datetime.fromisoformat(str(values))


def concat(frames: List[DataFrame]):
    data = []
    index = []
    for frame in frames:
        data.extend(frame._data)
        index.extend(frame.index)
    df = DataFrame(data)
    df.index = Index(index)
    return df


def to_numeric(values, errors: str = "raise"):
    try:
        return float(values)
    except Exception:
        if errors == "coerce":
            try:
                return float(str(values))
            except Exception:
                return None
        raise


class SimpleRow:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


__all__ = [
    "DataFrame",
    "Index",
    "Series",
    "to_datetime",
    "concat",
    "to_numeric",
]
