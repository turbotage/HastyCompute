# Centerlines in a flow value

The flow viewer shows centerlines when the flow value (the dict with `"kind": "flow4d"`) has an optional entry
`"centerlines"`. Nothing else is needed server-side: the viewer looks at the value summary and fetches the pieces itself.

```
"centerlines": {
    "points":  f32 [N, 3]   voxel index coordinates (x, y, z), voxel centres at integers — the same convention as the
                            points of the "lines" values (flow_streamlines / flow_pathlines),
    "offsets": u32 [L + 1]  line l is points[offsets[l] : offsets[l + 1]], ordered along the vessel,
    "radius":  f32 [N]      optional, lumen radius in mm (reserved; not used for the measurement yet)
}
```

With centerlines the vessel tool places its analysis planes ⟂ to the centerline at the nearest line (any direction is
fine: planes are oriented so that the net flow is positive); without them it follows the vessel from the click along the
time-mean flow direction and re-centres on the lumen at every plane.
