class Event:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data


class EventBus:
    def __init__(self):
        self.listeners = []
        self.event_history = []

    def raise_event(self, event: Event):
        for service in self.listeners:
            service.raise_event(event)

    def raise_request(self, event: Event):
        value = None
        for service in self.listeners:
            answer = service.raise_request(event)
            if answer is not None:
                value = answer()

        return value

    def raise_result(self, event: Event):
        value = None
        for service in self.listeners:
            answer = service.raise_result(event)
            if answer is not None:
                value = answer

        return value

    def register(self, listener):
        listener.event_bus = self
        self.listeners.append(listener)
