import os
from tkinter import filedialog as fd
from skimage.io import imsave
from skimage.util import img_as_ubyte
from anglemanagerv2 import AngleManager

HTML_HEADER = """<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN"
                        "http://www.w3.org/TR/html4/strict.dtd">
                    <html lang="en">
                      <head>
                        <meta http-equiv="content-type" content="text/html; charset=utf-8">
                        <title>title</title>
                        <link rel="stylesheet" type="text/css" href="style.css">
                        <script type="text/javascript" src="script.js"></script>
                      </head>
                      <body>\n"""

root_folder = fd.askdirectory()

report = HTML_HEADER
report += '<table cellpading=10 cellspacing=10>\n<th>Cell Name</th><th>Line1</th><th>Kym1</th><th>Kym1 Filtered</th><th>Kym1 Angle</th>'
report += '<th>Line2</th><th>Kym2</th><th>Kym2 Filtered</th><th>Kym2 Angle</th><th>Angle Difference</th>\n'
for cell in os.listdir(root_folder):

    working_path = root_folder + os.sep + cell
    app = AngleManager()
    app.load_kymographs(path_kym1=working_path+os.sep+"kym1.tif", path_kym2=working_path+os.sep+"kym2.tif")
    print(cell)
    app.compute_coords(method="PCA")
    app.compute_regression()
    app.compute_angles()

    imsave(working_path+os.sep+"kym1_w_line.png", img_as_ubyte(app.kymograph_1_w_line))
    imsave(working_path+os.sep+"kym2_w_line.png", img_as_ubyte(app.kymograph_2_w_line))
    imsave(working_path+os.sep+"kym1_filtered.png", img_as_ubyte(app.filtered_kym1))
    imsave(working_path+os.sep+"kym2_filtered.png", img_as_ubyte(app.filtered_kym2))

    report += '<tr><td valign="middle" width="100" align="center">' + cell + '</td><td valign="middle"  align="center"><img src="./' + cell + '/kymline1.png" alt="pic" width="200"/></td>'
    report += '<td valign="middle" align="center"><img src="./' + cell + '/kym1_w_line.png" alt="pic" width="200"/></td>'
    report += '<td valign="middle" align="center"><img src="./' + cell + '/kym1_filtered.png" alt="pic" width="200"/></td>'
    report += '<td valign="middle" width="100" align="center">' + ("{0:.2f}").format(app.kym1_angle) + '</td>'
    report += '<td valign="middle" align="center"><img src="./' + cell + '/kymline2.png" alt="pic" width="200"/></td>'
    report += '<td valign="middle" align="center"><img src="./' + cell + '/kym2_w_line.png" alt="pic" width="200"/></td>'
    report += '<td valign="middle" align="center"><img src="./' + cell + '/kym2_filtered.png" alt="pic" width="200"/></td>'
    report += '<td valign="middle" width="100" align="center">' + ("{0:.2f}").format(app.kym2_angle) + '</td>'
    report += '<td valign="middle" width="100" align="center">' + ("{0:.2f}").format(app.angle_diff) + '</td></tr>\n'

report += '</table></body>\n</html>'
open(root_folder + os.sep + "report.html", "w").write(report)

