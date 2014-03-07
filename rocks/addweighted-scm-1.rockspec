package = "addweighted"
version = "scm-1"

source = {
   url = "git://github.com/jolegrand/AddWeighted",
   branch = "master",
}

description = {
   summary = "torch module for nn",
   detailed = [[
   ]],
   homepage= "https://github.com/jlegrand/addWeighted",
   license = "GPL"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0"
}

build = {
   type = "builtin",
   modules = {
      ["addWeighted.init"] = "addWeighted.lua"
   }
}
