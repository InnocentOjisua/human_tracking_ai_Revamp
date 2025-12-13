import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="DPG Test", width=400, height=300)

with dpg.window(label="Hello Window"):
    dpg.add_text("🎉 DearPyGui is working!")
    dpg.add_button(label="Click Me")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
