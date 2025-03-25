graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "river"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "world-region"
  ]
  node [
    id 3
    label "person"
  ]
  node [
    id 4
    label "country"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 3
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 3
    target 4
    key 0
    label ":mod"
  ]
  edge [
    source 4
    target 1
    key 0
    label ":name"
  ]
]
