/* Flat-image layout for the milestone-2 gate.
 *
 * Everything the program needs is packed into one output section starting at
 * 0x1000 (the emulator's PROGRAM_BASE), so `objcopy -O binary` produces a file
 * whose byte 0 is the instruction the CPU will fetch first. Notes, unwind
 * tables and relocations are discarded: the image is already position-fixed.
 */
ENTRY(_start)
SECTIONS
{
    . = 0x1000;
    .text : {
        *(.text.entry)
        *(.text .text.*)
        *(.rodata .rodata.*)
        *(.data .data.*)
    }
    /DISCARD/ : {
        *(.comment)
        *(.note .note.*)
        *(.eh_frame .eh_frame_hdr)
        *(.rela .rela.*)
    }
}
