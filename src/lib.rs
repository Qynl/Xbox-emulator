//! `xone` — a from-scratch Xbox emulator, built in the open.
//!
//! The core is x86-64, which is the CPU architecture of the Xbox One family
//! (Durango, AMD Jaguar) and the Xbox Series S / X (Scarlett, AMD Zen 2), so one
//! core serves both generations. The 7th and 6th generations are PowerPC and
//! 32-bit x86 respectively and need cores of their own; see ROADMAP.md.
//!
//! This crate currently contains milestone 1: a real, executable x86-64 CPU
//! interpreter with a flat physical memory model. Everything here executes
//! actual machine code; nothing is stubbed to pretend an instruction works.
//!
//! The CPU instruction set is verified differentially against Unicorn (a
//! QEMU-derived x86 emulator) by `tools/diff_unicorn.py`, so regressions show
//! up as concrete register/flag/memory mismatches rather than vibes.

#![allow(dead_code)]

pub mod cpu;
pub mod flags;
pub mod memory;

pub use cpu::{Cpu, Stop};
