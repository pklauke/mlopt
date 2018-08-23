# !/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class TransformerMixin(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs).transform(*args, **kwargs)
