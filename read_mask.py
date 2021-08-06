import idlsave

# This saved mask file can be found at,
# /SNS/NOM/IPTS-24637/shared/autoNOM_051520/mask144025.dat
idl_mask_out = "mask144025.dat"
# This saved mask file can be found at,
# /SNS/NOM/IPTS-24637/shared/autoNOM_SiO2/mask144974.dat
# idl_mask_out = "mask144974.dat"

mask_idl = idlsave.read(idl_mask_out)
mask_out = open(idl_mask_out.replace(".dat", ".out"), "w")
for item in mask_idl.mask:
    mask_out.write("{0:10d}\n".format(int(item)))
mask_out.close()

