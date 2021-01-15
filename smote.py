import weka.core.jvm as jvm
import weka.core.packages as packages
from weka.core.classes import complete_classname

jvm.start(packages=True)

pkg = "SMOTE"

# install package if necessary
if not packages.is_installed(pkg):
    print("Installing %s..." % pkg)
    packages.install_package(pkg)
    print("Installed %s, please re-run script!" % pkg)
    jvm.stop()

# testing classname completion
print(complete_classname(".SMOTE"))

jvm.stop()
