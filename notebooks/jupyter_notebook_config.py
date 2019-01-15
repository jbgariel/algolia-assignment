c = get_config()  # get the config object
c.NotebookApp.port = 9999
c.IPKernelApp.pylab = 'inline'  # in-line figure when using Matplotlib
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False  # do not open a browser window by default when using notebooks
c.NotebookApp.token = ''  # No token. Always use jupyter over ssh tunnel
c.NotebookApp.notebook_dir = '/usr/src/app/notebooks/'
c.NotebookApp.allow_root = True  # Allow to run Jupyter from root user inside Docker container
