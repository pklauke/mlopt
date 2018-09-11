# !/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class TransformerMixin(ABC):
    """Abstract transformer mixin class."""

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the transformer on given data."""
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        """Transform the given data."""
        pass

    @abstractmethod
    def fit_transform(self, *args, **kwargs):
        """Fit and transform the given data."""
        self.fit(*args, **kwargs).transform(*args, **kwargs)
