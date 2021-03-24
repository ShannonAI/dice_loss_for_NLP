#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# file: tokenizer.py
#

import unicodedata

def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


if __name__ == "__main__":
    text = "俄 罗 斯 太 空 管 制 中 心 发 言 人 在 电 话 中 表 示 ： “ 和 平 号 ” 太 空 站 已 经 作 好 了 准 备 ， 预 计 ２ ７ 号 和 “ 进 展 号 ” 太 空 船 连 接 ， “ 进 展 号 ” 是 在 莫 斯 科 时 间 ２ ４ 号 上 午 ７ 点 ２ ９ 分 发 射 ， 这 次 的 发 射 原 定 在 １ ８ 号 进 行 ， 但 是 因 为 “ 和 平 号 ” 太 空 站 的 定 位 系 统 发 生 了 问 题 ， 因 此 俄 国 太 空 中 心 主 管 在 最 后 一 刻 决 定 要 延 后 发 射 。"
    print(f"ORIGIN -> {text}")
    re_text = _clean_text(text)
    print(f"AFTER -> {re_text}")