"""API for data_analysis"""

import dataclasses
from typing import Union, Dict


@dataclasses.dataclass
class AnalyzeRequest:
  query: str
  data: str | None = None
  data_requirement: str | None = None
  files: Union[list["FileMetadata"], None] = None


@dataclasses.dataclass
class AnalyzeResponse:
  execution_output: str | None = None
  generated_code: str | None = None
  generated_images: list[str] | None = None


@dataclasses.dataclass
class FileMetadata:
  filename: str | None = None
  source: str | None = None


def analyze(
    query: str,
    data: str | None = None,
    data_requirement: str | None = None,
    files: list[FileMetadata] | None = None,
) -> AnalyzeResponse:
  ...
