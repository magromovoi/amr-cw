graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "date-interval"
  ]
  node [
    id 1
    label "date-entity"
  ]
  node [
    id 2
    label "until"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":op1"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":op1"
  ]
]
