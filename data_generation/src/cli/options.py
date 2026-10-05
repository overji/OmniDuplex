"""Translate parsed command arguments into application inputs."""
from dataclasses import fields


def application_options(kind, args):
    return kind(**{field.name: getattr(args, field.name) for field in fields(kind)
                   if hasattr(args, field.name)})
