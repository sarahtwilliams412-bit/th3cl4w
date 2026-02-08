"""
Self-Filter (Arm Volume Subtraction) — Agent 4

Uses the D1 arm's current joint angles and known geometry to compute
which voxels belong to the arm itself. Subtracts them to produce an
obstacle-only grid and computes the Euclidean distance field.
"""
