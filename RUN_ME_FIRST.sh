#!/bin/bash
# Moneyball Dojo — Sheets接続テスト
# ターミナルで以下を実行してください:
#
#   cd moneyball_dojo
#   pip install gspread
#   python test_sheets_connection.py
#
# 「全テスト合格」と出たらOKです。

cd "$(dirname "$0")"
pip install gspread
python test_sheets_connection.py
