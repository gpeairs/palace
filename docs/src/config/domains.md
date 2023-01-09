```@raw html
<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. --->
<!--- SPDX-License-Identifier: Apache-2.0 --->
```

# `config["Domains"]`

```json
"Domains":
{
    "Materials":
    [
        ...
    ],
    "Postprocessing":
    {
        "Dielectric":
        [
            ...
        ],
        "Probe":
        [
            ...
        ]
    }
}
```

with

`"Materials"` :  Array of material properties objects.

`"Postprocessing"` :  Top-level object for configuring domain postprocessing.

`"Dielectric"` :  Array of objects for postprocessing bulk dielectric loss.

`"Probe"` :  Array of objects for postprocessing solution field values evaluated at a probe
location in space.

## `domains["Materials"]`

```json
"Materials":
[
    // Material 1
    {
        "Attributes": [<int array>],
        "Permeability": <float or float array>,
        "Permittivity": <float or float array>,
        "LossTan": <float or float array>,
        "Conductivity": <float or float array>,
        "LondonDepth": <float>,
        "MaterialAxes": <array of float array>
    },
    // Material 2, 3, ...
    ...
]
```

with

`"Attributes" [None]` :  Integer array of mesh domain attributes for this material.

`"Permeability" [1.0]` :  Relative permeability for this material. Scalar or
vector of 3 coefficients corresponding to each of `"MaterialAxes"`.

`"Permittivity" [1.0]` : Relative permittivity for this material. Scalar or
vector of 3 coefficients corresponding to each of `"MaterialAxes"`.

`"LossTan" [0.0]` :  Loss tangent for this material. Scalar or
vector of 3 coefficients corresponding to each of `"MaterialAxes"`.

`"Conductivity" [0.0]` :  Electrical conductivity for this material, S/m. Activates Ohmic
loss model in the material domain. Scalar or
vector of 3 coefficients corresponding to each of `"MaterialAxes"`.

`"LondonDepth" [0.0]` :  London penetration depth for this material, specified in mesh
length units. Activates London equations-based model relating superconducting current and
electromagnetic fields in the material domain.

`"MaterialAxes" [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]` : Axes directions
for specification of anisotropic material properties. Required to be unit length
and orthogonal.

## `domains["Postprocessing"]["Dielectric"]`

```json
"Postprocessing":
{
    "Dielectric":
    [
        {
            "Index": <int>,
            "Attributes": [<int array>]
        },
        ...
    ]
}
```

with

`"Index" [None]` :  Index of this lossy domain, used in postprocessing output files.

`"Attributes" [None]` :  Integer array of mesh domain attributes for this lossy domain.

## `domains["Postprocessing"]["Probe"]`

```json
"Postprocessing":
{
    "Probe":
    [
        {
            "Index": <int>,
            "X": <float>,
            "Y": <float>,
            "Z": <float>
        },
        ...
    ]
}
```

with

`"Index" [None]` :  Index of this probe, used in postprocessing output files.

`"X" [None]` :  ``x``-coordinate of this probe, specified in mesh length units.

`"Y" [None]` :  ``y``-coordinate of this probe, specified in mesh length units.

`"Z" [None]` :  ``z``-coordinate of this probe, specified in mesh length units.