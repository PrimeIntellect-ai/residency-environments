FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg2 wget sudo kmod \
    libvulkan1 vulkan-tools mesa-vulkan-drivers \
    libglib2.0-0 libgl1 libegl1 \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 ships a Vulkan loader too old for current NVIDIA drivers.
RUN curl -fsSL https://packages.lunarg.com/lunarg-signing-key-pub.asc \
    | gpg --dearmor -o /usr/share/keyrings/lunarg.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/lunarg.gpg] https://packages.lunarg.com/vulkan jammy main' \
    > /etc/apt/sources.list.d/lunarg-vulkan.list \
    && apt-get update && apt-get install -y --no-install-recommends libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Unreal refuses to run as root.
RUN useradd -m airsim_user

ADD CityParkLite_Linux.tar.gz /opt/cosys-airsim-city-park/
RUN chown -R airsim_user:airsim_user /opt/cosys-airsim-city-park \
    && find /opt/cosys-airsim-city-park -name '*.sh' -exec chmod +x {} +

RUN pip3 install --no-cache-dir cosysairsim==3.3.0

COPY aerialsim_flight_pkg/aerialsim_flight/drone.py /usr/lib/python3/dist-packages/aerialsim_flight/drone.py
RUN touch /usr/lib/python3/dist-packages/aerialsim_flight/__init__.py

COPY settings.json /home/airsim_user/Documents/AirSim/settings.json
RUN chown -R airsim_user:airsim_user /home/airsim_user/Documents

COPY start_sim.sh /usr/local/bin/start_sim.sh
RUN chmod +x /usr/local/bin/start_sim.sh

CMD ["/usr/local/bin/start_sim.sh"]
