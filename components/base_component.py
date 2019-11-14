class BaseComponent:
    """
    Template for component.    
    """

    def render(self, props={}):
        """
        Override and return Dash component created here.
        """
        pass

    def register_callbacks(self, app):
        """
        Override and register all callbacks here
        """
        pass