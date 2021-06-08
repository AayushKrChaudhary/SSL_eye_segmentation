#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ritnet import RITNet2D
model_dict = {}

model_dict['ritnet'] = RITNet2D(dropout=True,prob=0.2)

